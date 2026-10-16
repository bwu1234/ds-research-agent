"""The run viewer, offline: read-only ledger access, trace arrangement, and
the pages served from a synthetic ledger (no benchmark data, no model)."""

from __future__ import annotations

import http.client
import sqlite3
import threading
import urllib.parse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.agent.tools import TOOLS
from ds_research_agent.ledger import Ledger, LedgerError, Program, Step
from eval.kramabench import report
from eval.kramabench.run import new_batch
from eval.kramabench.viewer import Context, ViewerServer, trace
from eval.kramabench.viewer.markup import esc, tag
from eval.kramabench.viewer.pages import headline
from eval.kramabench.viewer.server import CSP
from tests.test_given_files import _harness, _run

TOOL_JSON = [
    {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
    }
    for t in TOOLS
]


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": args}}


def _step(seq: int, messages: list[dict[str, Any]], reply: dict[str, Any] | None) -> Step:
    response = None
    if reply is not None:
        response = {
            "message": {
                "role": "assistant",
                "content": reply.get("content", ""),
                "thinking": None,
                "tool_calls": [
                    {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                    for c in reply.get("tool_calls", [])
                ],
                "tool_name": None,
            },
            "usage": {},
            "done_reason": "stop",
            "wall_s": 1.0,
        }
    return Step(
        run_id="b/t/0",
        seq=seq,
        kind="model",
        request_sha256="x",
        request={"messages": messages, "tools": TOOL_JSON},
        response=response,
        wall_s=1.0,
        error=None if reply is not None else "ModelResponseError: boom",
    )


def _cell(seq: int, step: int, index: int, code: str) -> Program:
    return Program(
        run_id="b/t/0",
        seq=seq,
        step=step,
        call_index=index,
        kind="cell",
        session="s",
        code=code,
        status="ok",
        wall_s=0.1,
        audit_complete=True,
        observed_reads=[],
        record={"stdout": "1\n"},
    )


# ---- ledger ------------------------------------------------------------------


def test_readonly_ledger_refuses_writes_and_never_creates(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    with pytest.raises(LedgerError, match="no ledger"):
        Ledger(path, readonly=True)
    assert not path.exists()
    sqlite3.connect(path).close()  # an empty database: no schema, not created by the viewer
    with pytest.raises(LedgerError, match="schema None"):
        Ledger(path, readonly=True)
    path.unlink()
    Ledger(path).close()  # a real ledger
    ro = Ledger(path, readonly=True)
    assert ro.batches() == []
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro._db.execute("CREATE TABLE x (a)")


# ---- trace -------------------------------------------------------------------


def test_trace_numbers_loop_steps_through_a_repaired_request() -> None:
    system = {"role": "system", "content": "sys"}
    user = {"role": "user", "content": "Question: q"}
    bad = {"role": "assistant", "content": "", "tool_calls": [_call("run_python", {})]}
    fix = {"role": "tool", "tool_name": "run_python", "content": "Error: missing code"}
    good = {"role": "assistant", "content": "", "tool_calls": [_call("run_python", {"code": "1"})]}
    result = {"role": "tool", "tool_name": "run_python", "content": "[status: ok]"}
    done = {"role": "assistant", "content": "done"}
    m0 = [system, user]
    m1 = m0 + [bad, fix]
    m2 = m1 + [good, result]
    steps = [
        _step(0, m0, bad),  # rejected: run_python without code
        _step(1, m1, None),  # a failed request, same loop step
        _step(2, m1, good),  # accepted: loop step 1
        _step(3, m2, done),  # loop step 2
    ]
    tr = trace.build(steps, [_cell(0, 1, 0, "1"), _cell(1, 9, 0, "lost")])
    assert (
        tr.append_only
        and tr.system == "sys"
        and [m.content for m in tr.preamble] == ["Question: q"]
    )
    assert [(v.loop_step, v.accepted) for v in tr.steps] == [
        (1, False),
        (1, False),
        (1, True),
        (2, True),
    ]
    # Each request shows what it added: its reply's context copy, then results.
    assert [m.role for m in tr.steps[0].messages] == ["assistant", "tool"]
    assert tr.steps[1].messages == []
    placed = tr.steps[2].messages[0].calls[0]
    assert placed.code == "1" and [p.seq for p in placed.programs] == [0]
    assert [m.content for m in tr.steps[3].messages] == ["done"]
    assert [p.seq for p in tr.unplaced] == [1]  # listed, not dropped
    assert tr.tools == [t.name for t in TOOLS] and tr.tool_specs == TOOL_JSON


def test_trace_falls_back_when_requests_do_not_extend_each_other() -> None:
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    b = [a[0], {"role": "user", "content": "rewritten"}]
    tr = trace.build([_step(0, a, {"content": "x"}), _step(1, b, {"content": "y"})], [])
    assert not tr.append_only
    assert [m.content for m in tr.steps[1].messages] == ["rewritten", "y"]


def test_markup_escapes_text_and_attributes() -> None:
    out = tag("a", "<script>x</script>", href='/r?id="><img>', name="n")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert 'href="/r?id=&quot;&gt;&lt;img&gt;"' in out and 'name="n"' in out


# ---- pages -------------------------------------------------------------------

EVIL = "<script>alert(1)</script>"


@contextmanager
def _serving(tmp_path: Path, unseal: bool = False) -> Generator[tuple[ViewerServer, Any]]:
    h = _harness(tmp_path)
    keys = h.split.dev[:3]
    t = [h.tasks[k] for k in keys]
    batch = _run(h, keys, {t[0].query: t[0].gold.answer, t[1].query: EVIL, t[2].query: None}, {})
    sealed = new_batch(
        h.settings, h.model, h.split, condition="given_files", task_set="holdout",
        keys=h.split.holdout[:1], repeats=1, batch_id="sealed-batch",
    )  # fmt: skip
    h.ledger.add_batch(sealed)
    ctx = Context(h.settings, h.split, h.tasks, unseal_holdout=unseal)
    srv = ViewerServer(0, ctx, h.settings.eval.ledger_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv, (h, batch)
    finally:
        srv.shutdown()
        srv.server_close()


def _get(srv: ViewerServer, path: str, host: str | None = None) -> tuple[int, str, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
    conn.request("GET", path, headers={"Host": host or f"127.0.0.1:{srv.port}"})
    r = conn.getresponse()
    body = r.read().decode()
    conn.close()
    return r.status, body, r.headers


def _q(**kw: str) -> str:
    return urllib.parse.urlencode(kw)


def test_pages_render_runs_escape_text_and_seal_the_holdout(tmp_path: Path) -> None:
    with _serving(tmp_path) as (srv, (h, batch)):
        status, body, headers = _get(srv, "/")
        assert status == 200 and headers["Content-Security-Policy"] == CSP
        assert "agent-test" in body and "sealed-batch" not in body

        status, body, _ = _get(srv, f"/batch?{_q(id=batch.batch_id)}")
        assert status == 200 and "strict accuracy" in body and "budget_exhausted" not in body
        assert _get(srv, f"/batch?{_q(id='sealed-batch')}")[0] == 404

        runs = h.ledger.runs(batch.batch_id)
        status, body, _ = _get(srv, f"/run?{_q(id=runs[1].run_id)}")
        assert status == 200
        assert "ran as cell" in body and "submit_answer" in body and "verification" in body
        assert EVIL not in body and "&lt;script&gt;alert(1)&lt;/script&gt;" in body
        assert "Evaluator only: gold answer" in body
        # Tool definitions are shown in full, beside the system prompt.
        assert "tool definitions as sent · run_python, submit_answer" in body
        assert "system turn as the model read it" in body and "&lt;tools&gt;" in body
        assert esc(TOOLS[0].description.split(".")[0]) in body

        status, body, _ = _get(srv, f"/step?{_q(id=runs[1].run_id, seq='0')}")
        assert status == 200 and "&quot;tools&quot;" in body and "request sha256" in body
        assert 'id="prompt"' in body and "&lt;|im_start|&gt;system" in body
        assert _get(srv, f"/step?{_q(id=runs[1].run_id, seq='x')}")[0] == 404
        assert _get(srv, f"/step?{_q(id=runs[1].run_id, seq='99')}")[0] == 404

        status, body, _ = _get(srv, f"/task?{_q(key=runs[0].task_key)}")
        assert status == 200 and runs[0].run_id.replace("/", "%2F") in body
        assert _get(srv, f"/task?{_q(key=h.split.holdout[0])}")[0] == 404

        q = _q(base=batch.batch_id, other=batch.batch_id, subset="all")
        status, body, _ = _get(srv, f"/compare?{q}")
        assert status == 200 and "+0.000" in body

        assert _get(srv, "/static/viewer.js")[0] == 200
        assert _get(srv, "/static/../server.py")[0] == 404
        assert _get(srv, "/nope")[0] == 404
        assert _get(srv, f"/run?{_q(id='agent-test/none/0')}")[0] == 404
        # A page that rebinds its own domain to 127.0.0.1 is refused.
        assert _get(srv, "/", host=f"evil.example:{srv.port}")[0] == 403


def test_unsealed_viewer_shows_holdout_batches(tmp_path: Path) -> None:
    with _serving(tmp_path, unseal=True) as (srv, _):
        assert "sealed-batch" in _get(srv, "/")[1]
        assert _get(srv, f"/batch?{_q(id='sealed-batch')}")[0] == 200


def test_headline_matches_the_report(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    keys = h.split.dev[:3]
    t = [h.tasks[k] for k in keys]
    batch = _run(h, keys, {t[0].query: t[0].gold.answer, t[1].query: "x", t[2].query: None}, {})
    ctx = Context(h.settings, h.split, h.tasks)
    ro = Ledger(h.settings.eval.ledger_path, readonly=True)
    got = headline(ro, ctx, batch)
    want = report.summarise(h.ledger, batch, h.tasks, 50, 0)
    assert got["score"] == pytest.approx(want["answer_score"], abs=5e-5)
    assert got["strict"] == pytest.approx(want["strict_accuracy"], abs=5e-5)
    assert got["verified"] == pytest.approx(want["verified_success"]["rate"], abs=5e-5)
    assert (got["recorded"], got["planned"]) == (want["runs_recorded"], want["runs_planned"])
