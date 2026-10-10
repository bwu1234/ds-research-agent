"""The given-files condition in the KramaBench harness, offline.

A scripted model and a fake workspace run synthetic tasks end to end:
ledger rows, scoring with verified success, the report, and replay of
recorded agent runs with no model and no sandbox.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.ledger import Ledger
from ds_research_agent.models import ChatMessage, ChatResult, ToolCall, ToolSpec, Usage
from ds_research_agent.sandbox import InputMount
from eval.kramabench import agent_runs, failures, report
from eval.kramabench.baselines import Harness, ReplayClient, ReplayMismatch, run_batch
from eval.kramabench.run import new_batch
from eval.kramabench.scoring import PROFILE
from eval.kramabench.split import split_for
from eval.kramabench.tasks import Task
from tests.kramabench_synthetic import settings_for
from tests.test_agent_loop import FakeWorkspace, rerun


def _reply(name: str, args: dict[str, Any]) -> ChatResult:
    return ChatResult(
        message=ChatMessage(role="assistant", tool_calls=(ToolCall(name=name, arguments=args),)),
        usage=Usage(prompt_eval_tokens=100, output_tokens=20, eval_ns=10**9, prompt_eval_ns=10**9),
        done_reason="stop",
        wall_s=2.0,
    )


class AgentScript:
    """Explores once, then submits ``answers[question]`` with the listed files.

    Questions mapped to None never submit (the run hits the step budget)."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        user = messages[1].content
        question = user.splitlines()[0].removeprefix("Question: ")
        files = re.findall(r"^- (\S+) \(", user, re.M)
        answer = self.answers[question]
        if len(messages) == 2 or answer is None:
            return _reply("run_python", {"code": "print(1)"})
        program = f"print({json.dumps(json.dumps({'answer': answer}))})"
        return _reply("submit_answer", {"answer": answer, "files_used": files, "program": program})


def _harness(tmp_path: Path) -> Harness:
    s = settings_for(tmp_path)
    split, tasks = split_for(s)
    s = s.model_copy(update={"eval": s.eval.model_copy(update={"split_sha256": split.sha256()})})
    return Harness(s, s.model, Ledger(s.eval.ledger_path), split, tasks)


def _workspace_for(reads: dict[str, bool]):  # type: ignore[no-untyped-def]
    """Reruns print the submitted answer; ``reads[key]`` says whether they read the files."""
    current: dict[str, Any] = {}

    def make(t: Task, _r: int, mounts: Sequence[InputMount]) -> FakeWorkspace:
        paths = [m.container_path for m in mounts] if reads.get(t.key, True) else []
        ws = FakeWorkspace(final=rerun("", reads=paths))
        current[t.key] = ws

        async def rerun_prints(program: str) -> Any:  # echo the program's literal
            text = json.loads(program.removeprefix("print(").removesuffix(")"))
            return rerun(text, reads=paths)

        ws.rerun = rerun_prints  # type: ignore[method-assign]
        return ws

    return make


def _run(h: Harness, keys: tuple[str, ...], answers: dict[str, Any], reads: dict[str, bool]):  # type: ignore[no-untyped-def]
    batch = new_batch(
        h.settings, h.model, h.split, condition="given_files", task_set="dev", keys=keys,
        repeats=1, batch_id="agent-test",
    )  # fmt: skip
    h.ledger.add_batch(batch)
    client = AgentScript(answers)
    ws_for = _workspace_for(reads)

    async def task(hh: Harness, b: Any, t: Task, r: int, c: Any) -> Any:
        return await agent_runs.run_agent_task(hh, b, t, r, c, ws_for, "sha256:img")

    asyncio.run(run_batch(h, batch, lambda _t, _r: client, agent_task=task))
    return batch


def test_given_files_runs_score_verify_report_and_replay(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    keys = h.split.dev[:5]
    t = [h.tasks[k] for k in keys]
    answers = {
        t[0].query: t[0].gold.answer,  # right, verified
        t[1].query: "wrong",  # wrong: reproduces, not strict
        t[2].query: t[2].gold.answer,  # right, but the rerun reads nothing
        t[3].query: None,  # never submits
        t[4].query: t[4].gold.answer,
    }
    batch = _run(h, keys, answers, reads={keys[2]: False})
    runs = {r.task_key: r for r in h.ledger.runs(batch.batch_id)}
    stop = [runs[k].stop_reason for k in keys]
    assert stop == ["submitted", "submitted", "submitted", "max_steps", "submitted"]

    r0 = runs[keys[0]]
    m = r0.input_manifest
    assert r0.access_policy == "labelled_files_mounted" and m["sandbox_image"] == "sha256:img"
    assert m["files"][0]["path"].startswith(f"/data/{t[0].domain}/input/")
    assert m["files"][0]["sha256"] and m["answer_type_visible"] is False
    steps = h.ledger.steps(r0.run_id)
    assert len(steps) == 2 and steps[0].request["tools"][0]["function"]["name"] == "run_python"
    assert "Expected answer type" not in steps[0].request["messages"][1]["content"]
    assert [p.kind for p in h.ledger.programs(r0.run_id)] == ["cell", "final_rerun"]
    assert batch.system_prompt_sha256 == agent_runs.prefix_sha256()

    verified = {
        k: (s.strict, s.verified_success)
        for k in keys
        if (s := h.ledger.score(runs[k].run_id, PROFILE)) is not None
    }
    assert verified[keys[0]] == (True, True)
    assert verified[keys[1]] == (False, False)
    assert verified[keys[2]] == (True, False)  # right answer, access not verified
    assert verified[keys[3]] == (False, False)
    assert h.ledger.answer(runs[keys[3]].run_id).parse_status == "max_steps"  # type: ignore[union-attr]

    out = report.summarise(h.ledger, batch, h.tasks, 200, 0)
    assert out["verified_success"]["rate"] == pytest.approx(2 / 5)
    assert out["agent"]["submitted"] == 4 and out["agent"]["access_verified"] == 3
    assert out["stop_reasons"] == {"submitted": 4, "max_steps": 1}

    # Replay: recorded responses and programs, no model, no sandbox.
    replay = new_batch(
        h.settings, h.model, h.split, condition="given_files", task_set="dev", keys=keys,
        repeats=1, batch_id="agent-replay", replay_of=batch,
    )  # fmt: skip
    h.ledger.add_batch(replay)
    steps_of = {k: h.ledger.steps(runs[k].run_id) for k in keys}
    progs_of = {k: h.ledger.programs(runs[k].run_id) for k in keys}

    async def task(hh: Harness, b: Any, tk: Task, r: int, c: Any) -> Any:
        ws = agent_runs.ReplayWorkspace(progs_of[tk.key])
        return await agent_runs.run_agent_task(hh, b, tk, r, c, lambda *_: ws, "sha256:img")

    asyncio.run(
        run_batch(
            h, replay, lambda tk, _r: ReplayClient(h.model, steps_of[tk.key]), agent_task=task
        )
    )
    for a, b in zip(h.ledger.runs(batch.batch_id), h.ledger.runs("agent-replay"), strict=True):
        assert (a.stop_reason, a.error) == (b.stop_reason, b.error)
        sa, sb = h.ledger.score(a.run_id, PROFILE), h.ledger.score(b.run_id, PROFILE)
        assert sa and sb and (sa.score, sa.strict, sa.verified_success) == (
            sb.score, sb.strict, sb.verified_success,
        )  # fmt: skip
        va, vb = h.ledger.verification(a.run_id), h.ledger.verification(b.run_id)
        assert (va and va.model_dump(exclude={"run_id"})) == (
            vb and vb.model_dump(exclude={"run_id"})
        )


def test_replay_workspace_refuses_changed_code(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    keys = h.split.dev[:1]
    task = h.tasks[keys[0]]
    batch = _run(h, keys, {task.query: task.gold.answer}, {})
    progs = h.ledger.programs(h.ledger.runs(batch.batch_id)[0].run_id)
    ws = agent_runs.ReplayWorkspace(progs)
    with pytest.raises(ReplayMismatch):
        asyncio.run(ws.execute("print(2)", 5))


def test_failure_labels_rules_manual_counts_and_render(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    keys = h.split.dev[:3]
    t = [h.tasks[k] for k in keys]
    answers = {t[0].query: t[0].gold.answer, t[1].query: "wrong", t[2].query: None}
    batch = _run(h, keys, answers, {})
    runs = {r.task_key: r for r in h.ledger.runs(batch.batch_id)}
    assert failures.apply_rules(h.ledger, batch.batch_id) == 1
    lab = h.ledger.failure_label(runs[keys[2]].run_id)
    assert lab and (lab.category, lab.source) == ("budget_exhausted", "rule")
    assert h.ledger.failure_label(runs[keys[1]].run_id) is None  # needs a reader
    with pytest.raises(ValueError, match="strictly correct"):
        failures.label(h.ledger, runs[keys[0]], "other", "x")
    with pytest.raises(ValueError, match="unknown category"):
        failures.label(h.ledger, runs[keys[1]], "typo", "x")
    failures.label(h.ledger, runs[keys[1]], "wrong_filter_or_join", "read")
    # A manual label is never overwritten by a rule.
    failures.label(h.ledger, runs[keys[2]], "other", "looked closer")
    failures.apply_rules(h.ledger, batch.batch_id)
    assert h.ledger.failure_label(runs[keys[2]].run_id).category == "other"  # type: ignore[union-attr]

    out = report.summarise(h.ledger, batch, h.tasks, 200, 0)["failures"]
    assert out["failed_runs"] == 2 and out["unlabelled"] == 0
    assert out["counts"] == {"wrong_filter_or_join": 1, "other": 1}

    text = failures.render_run(h.ledger, runs[keys[1]], t[1])
    assert "gold (evaluator only)" in text and "> submit_answer" in text
    assert "## verification" in text and "failure label: wrong_filter_or_join" in text
