"""Offline checks of the agent loop with a scripted model and a fake workspace."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.agent import (
    SYSTEM_PROMPT,
    TOOLS,
    Outcome,
    ProgramEvent,
    Submission,
    Verification,
    run_agent,
    user_prompt,
)
from ds_research_agent.agent.answers import check_submission, program_answer, reproduces
from ds_research_agent.agent.loop import render_cell
from ds_research_agent.agent.tools import BUDGET_LOW, LAST_TURN, SANDBOX_PACKAGES, TRUNCATED
from ds_research_agent.config import AgentSettings
from ds_research_agent.models import (
    ChatMessage,
    ChatResult,
    ModelResponseError,
    ToolCall,
    ToolSpec,
    Usage,
)
from ds_research_agent.sandbox import (
    AuditResult,
    CellResult,
    ObservedRead,
    SandboxError,
    SandboxRun,
    SandboxUnavailable,
    SessionEnd,
)

F = "/data/legal/input/a.csv"
G = "/data/legal/input/b.csv"
SETTINGS = AgentSettings(
    tool_call_max_repairs=1,
    max_steps=6,
    max_wall_s=1000,
    max_output_tokens=4096,
    max_tool_output_chars=50,
    replan_after_repeats=2,
    budget_warning_steps=0,
    submit_checks=0,
    server_retries=1,
)


def audit(reads: Sequence[str] = (), complete: bool = True) -> AuditResult:
    return AuditResult(
        complete=complete,
        issues=[] if complete else ["timed out"],
        processes=1,
        data_reads=sorted(reads),
        data_dirs_listed=[],
        failed_data_opens=[],
        blocked_escapes=[],
        unparsed_lines=0,
    )


def cell(status: str = "ok", stdout: str = "", tb: str | None = None, **kw: Any) -> CellResult:
    return CellResult(
        cell=1,
        code="",
        status=status,  # type: ignore[arg-type]
        traceback=tb,
        stdout=stdout,
        stderr=kw.get("stderr", ""),
        stdout_truncated=kw.get("stdout_truncated", False),
        stderr_truncated=False,
        wall_s=0.01,
        audit=audit(),
        observed_reads=[],
    )


def rerun(  # type: ignore[no-untyped-def]
    stdout: str,
    reads: Sequence[str] = (F,),
    exit_code: int = 0,
    complete: bool = True,
    stderr: str = "",
):
    return SandboxRun(
        container="dsra-sbx-test",
        image_id="sha256:x",
        exit_code=exit_code,
        timed_out=False,
        wall_s=0.5,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=False,
        stderr_truncated=False,
        audit=audit(reads, complete),
        observed_reads=[ObservedRead(container_path=r, host_path=Path("/h") / r) for r in reads],
        scratch_dir=Path("/s"),
    )


def reply(*calls: tuple[str, dict[str, Any]], content: str = "", done: str = "stop") -> ChatResult:
    return ChatResult(
        message=ChatMessage(
            role="assistant",
            content=content,
            tool_calls=tuple(ToolCall(name=n, arguments=a) for n, a in calls),
        ),
        usage=Usage(),
        done_reason=done,
        wall_s=1.0,
    )


def submit(answer: Any = 3, files: Sequence[str] = (F,), program: str = "print(1)"):  # type: ignore[no-untyped-def]
    return ("submit_answer", {"answer": answer, "files_used": list(files), "program": program})


def py(code: str = "x = 1") -> tuple[str, dict[str, Any]]:
    return ("run_python", {"code": code})


class Script:
    """A model that returns scripted replies and records each request."""

    def __init__(self, *replies: ChatResult | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        assert tuple(tools) == TOOLS
        self.requests.append(list(messages))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeWorkspace:
    def __init__(
        self,
        cells: Sequence[CellResult | Exception] = (),
        final: SandboxRun | None = None,
        failing: Sequence[SandboxRun] = (),
    ) -> None:
        self.cells = list(cells)
        self.failing = list(failing)  # returned by the first reruns, in order
        self.final = final or rerun('{"answer": 3}')
        self.executed: list[tuple[str, int]] = []
        self.reruns: list[str] = []
        self.closed = False

    @property
    def session_id(self) -> str | None:
        return "dsra-krn-test"

    async def execute(self, code: str, timeout_s: int) -> CellResult:
        self.executed.append((code, timeout_s))
        item = self.cells.pop(0) if self.cells else cell()
        if isinstance(item, Exception):
            raise item
        return item

    async def rerun(self, program: str) -> SandboxRun:
        self.reruns.append(program)
        return self.failing.pop(0) if self.failing else self.final

    async def close(self) -> list[SessionEnd]:
        self.closed = True
        return []


class Rec:
    def __init__(self) -> None:
        self.programs: list[ProgramEvent] = []
        self.submissions: list[tuple[Submission, int]] = []
        self.verifications: list[Verification] = []

    def program(self, event: ProgramEvent) -> None:
        self.programs.append(event)

    def submission(self, submission: Submission, step: int) -> None:
        self.submissions.append((submission, step))

    def verification(self, verification: Verification) -> None:
        self.verifications.append(verification)


async def go(
    client: Script, ws: FakeWorkspace, rec: Rec | None = None, **kw: Any
) -> tuple[Outcome, Rec]:
    rec = rec or Rec()
    out = await run_agent(
        client,
        ws,
        rec,
        system=SYSTEM_PROMPT,
        user=user_prompt("How many?", [(F, 10), (G, 20)], None),
        files=[F, G],
        settings=kw.pop("settings", SETTINGS),
        cell_timeout_s=30,
        **kw,
    )
    return out, rec


def last_tool_text(client: Script, i: int = -1) -> str:
    return client.requests[i][-1].content


# --- the loop --------------------------------------------------------------


async def test_explore_then_submit_reproduces_and_verifies_access() -> None:
    client = Script(reply(py("import pandas")), reply(submit()))
    ws = FakeWorkspace([cell(stdout="5\n")])
    out, rec = await go(client, ws)
    assert out.stop_reason == "submitted" and out.steps == 2
    assert ws.executed == [("import pandas", 30)] and ws.reruns == ["print(1)"] and ws.closed
    # The second request carries the cell result as a tool message.
    assert client.requests[1][-1].role == "tool"
    assert last_tool_text(client, 1) == "[status: ok]\nstdout:\n5\n"
    v = out.verification
    assert v is not None and v.reproduced and v.access_verified and v.detail == ()
    assert [p.kind for p in rec.programs] == ["cell", "final_rerun"]
    assert rec.submissions[0][1] == 2 and rec.verifications == [v]


async def test_request_prefix_is_append_only() -> None:
    client = Script(reply(py()), reply(py()), reply(submit()))
    await go(client, FakeWorkspace())
    for a, b in itertools.pairwise(client.requests):
        assert b[: len(a)] == a


@pytest.mark.parametrize(
    ("final", "files", "why"),
    [
        (rerun('{"answer": 4}'), (F,), "differs from the submitted"),
        (rerun("no answer"), (F,), 'no {"answer"'),
        (rerun('{"answer": 3}', exit_code=1), (F,), "exited 1"),
    ],
)
async def test_not_reproduced(final: SandboxRun, files: Sequence[str], why: str) -> None:
    out, _ = await go(Script(reply(submit(files=files))), FakeWorkspace(final=final))
    v = out.verification
    assert v is not None and not v.reproduced and any(why in d for d in v.detail)


CRASH = rerun("", exit_code=1, stderr="Traceback...\nNameError: name 'json' is not defined")
CHECKED = SETTINGS.model_copy(update={"submit_checks": 2})


async def test_failing_program_is_returned_at_submit_and_can_be_fixed() -> None:
    client = Script(reply(submit(program="bad")), reply(submit(program="good")))
    ws = FakeWorkspace(failing=[CRASH])
    out, rec = await go(client, ws, settings=CHECKED)
    text = last_tool_text(client, 1)
    assert text.startswith("Error: your program was run in a fresh sandbox and exited with code 1")
    assert "Nothing was submitted" in text and "NameError: name 'json'" in text
    assert out.stop_reason == "submitted" and out.steps == 2 and ws.reruns == ["bad", "good"]
    assert [(p.kind, p.code) for p in rec.programs] == [
        ("submit_check", "bad"),
        ("final_rerun", "good"),
    ]
    assert len(rec.submissions) == 1 and out.verification and out.verification.reproduced


async def test_missing_answer_line_is_returned_but_a_wrong_answer_is_not() -> None:
    client = Script(reply(submit()), reply(submit()))
    ws = FakeWorkspace(failing=[rerun("{'answer': 3}")], final=rerun('{"answer": 4}'))
    out, rec = await go(client, ws, settings=CHECKED)
    assert 'printed no {"answer": ...} JSON as its last line' in last_tool_text(client, 1)
    # The mismatch is not revealed: the second submission is final.
    assert out.steps == 2 and out.verification and not out.verification.reproduced
    assert [p.kind for p in rec.programs] == ["submit_check", "final_rerun"]


async def test_submit_checks_are_bounded_and_the_last_step_is_final() -> None:
    one = CHECKED.model_copy(update={"submit_checks": 1})
    client = Script(reply(submit()), reply(submit()))
    out, rec = await go(client, FakeWorkspace(failing=[CRASH, CRASH]), settings=one)
    assert out.steps == 2 and out.stop_reason == "submitted"
    assert [p.kind for p in rec.programs] == ["submit_check", "final_rerun"]
    assert out.verification and "rerun exited 1" in out.verification.detail
    last = CHECKED.model_copy(update={"max_steps": 2})
    client = Script(reply(py()), reply(submit()))
    out, rec = await go(client, FakeWorkspace(failing=[CRASH]), settings=last)
    assert out.stop_reason == "submitted" and out.submission is not None
    assert [p.kind for p in rec.programs] == ["cell", "final_rerun"]


async def test_access_needs_complete_audit_and_matching_claims() -> None:
    claims = Script(reply(submit(files=(F, G))))
    out, _ = await go(claims, FakeWorkspace())
    assert out.verification and out.verification.reproduced
    assert not out.verification.access_verified
    incomplete = FakeWorkspace(final=rerun('{"answer": 3}', complete=False))
    out, _ = await go(Script(reply(submit())), incomplete)
    assert out.verification and not out.verification.access_verified
    no_reads = FakeWorkspace(final=rerun('{"answer": 3}', reads=()))
    out, _ = await go(Script(reply(submit(files=()))), no_reads)
    assert out.verification and not out.verification.access_verified
    assert "rerun read no data files" in out.verification.detail


async def test_bad_submission_is_returned_and_can_be_fixed() -> None:
    client = Script(reply(submit(files=("/etc/passwd",))), reply(submit()))
    out, rec = await go(client, FakeWorkspace())
    assert "not given for this task: /etc/passwd" in last_tool_text(client, 1)
    assert out.stop_reason == "submitted" and len(rec.submissions) == 1


async def test_no_tool_call_and_cut_off_replies_are_nudged() -> None:
    client = Script(reply(content="thinking"), reply(content="x", done="length"), reply(submit()))
    out, _ = await go(client, FakeWorkspace())
    assert client.requests[1][-1].content.startswith("No tool was called")
    assert client.requests[2][-1].content.startswith("Your reply was cut off")
    assert out.stop_reason == "submitted" and out.steps == 3


async def test_step_budget_ends_without_an_answer() -> None:
    client = Script(*[reply(py()) for _ in range(6)])
    out, rec = await go(client, FakeWorkspace())
    assert out.stop_reason == "max_steps" and out.steps == 6
    assert out.submission is None and rec.submissions == []


async def test_budget_notice_on_the_last_steps_keeps_context_append_only() -> None:
    warned = SETTINGS.model_copy(update={"budget_warning_steps": 2})
    client = Script(*[reply(py()) for _ in range(4)], reply(content="hm"), reply(py()))
    out, _ = await go(client, FakeWorkspace(), settings=warned)
    assert out.stop_reason == "max_steps"
    texts = [last_tool_text(client, i) for i in range(1, 6)]
    assert not any("turns left" in t or "last turn" in t for t in texts[:3])
    assert texts[3].endswith(BUDGET_LOW.format(n=2))  # after step 4 of 6
    assert texts[4].startswith("No tool was called") and texts[4].endswith(LAST_TURN)
    for a, b in itertools.pairwise(client.requests):
        assert b[: len(a)] == a


def test_user_prompt_states_the_turn_budget() -> None:
    assert "turns" not in user_prompt("Q?", [(F, 1)], None)
    assert "You have 12 turns" in user_prompt("Q?", [(F, 1)], None, 12)


async def test_sandbox_failure_ends_the_run_but_docker_outage_propagates() -> None:
    ws = FakeWorkspace([SandboxError("container is gone")])
    out, _ = await go(Script(reply(py())), ws)
    assert out.stop_reason == "sandbox_error" and "container is gone" in (out.error or "")
    down = FakeWorkspace([SandboxUnavailable("Docker is unreachable")])
    with pytest.raises(SandboxUnavailable):
        await go(Script(reply(py())), down)
    assert down.closed


async def test_wall_budget_ends_the_run() -> None:
    ticks = iter([0.0, 0.0, 500.0, 1200.0, 1200.0, 1200.0])
    client = Script(reply(py()), reply(py()))
    out, _ = await go(client, FakeWorkspace(), clock=lambda: next(ticks))
    assert out.stop_reason == "max_wall" and out.steps == 1


async def test_cell_timeout_is_capped_by_wall_left() -> None:
    ticks = iter([0.0, 0.0, 990.0, 990.0, 2000.0, 2000.0])
    ws = FakeWorkspace()
    await go(Script(reply(py())), ws, clock=lambda: next(ticks))
    assert ws.executed[0][1] == 10


async def test_repeated_error_demands_a_replan() -> None:
    err = "Traceback...\nKeyError: 'col'\n"
    ws = FakeWorkspace(
        [cell("error", tb=err), cell("error", tb=err), cell("error", tb="ValueError: x")]
    )
    client = Script(reply(py()), reply(py()), reply(py()), reply(submit()))
    _, rec = await go(client, ws)
    assert "Before running more code" not in last_tool_text(client, 1)
    assert "occurred 2 times in a row" in last_tool_text(client, 2)
    assert "Before running more code" not in last_tool_text(client, 3)
    assert [p.replan_requested for p in rec.programs if p.kind == "cell"] == [False, True, False]


async def test_dead_kernel_is_reported() -> None:
    client = Script(reply(py()), reply(submit()))
    await go(client, FakeWorkspace([cell("dead")]))
    assert "kernel was restarted after the previous call (dead)" in last_tool_text(client, 1)


async def test_calls_after_submit_are_not_run() -> None:
    client = Script(reply(submit(), py("never")))
    ws = FakeWorkspace()
    out, _ = await go(client, ws)
    assert out.stop_reason == "submitted" and ws.executed == []


async def test_repair_exhaustion_stops_with_tool_call_failure() -> None:
    bad = reply(("drop_tables", {}))
    out, _ = await go(Script(bad, bad), FakeWorkspace())
    assert out.stop_reason == "tool_call_failure" and out.steps == 1


PANIC = ModelResponseError("mlx runner failed: panic: mlx: Maximum threads per threadgroup", 500)


async def test_runner_crash_is_resent_unchanged_not_repaired() -> None:
    client = Script(reply(py()), PANIC, reply(submit()))
    out, _ = await go(client, FakeWorkspace())
    assert out.stop_reason == "submitted"
    # The resend is byte-identical: no "rejected" feedback was appended.
    assert client.requests[1] == client.requests[2]
    assert not any("rejected" in m.content for m in client.requests[2])


async def test_runner_crash_beyond_retries_is_a_model_error() -> None:
    out, _ = await go(Script(PANIC, PANIC), FakeWorkspace())
    assert out.stop_reason == "model_error" and "runner failed" in (out.error or "")


async def test_parse_error_is_still_repaired() -> None:
    parse = ModelResponseError("XML syntax error on line 23: element <function> closed", 500)
    client = Script(parse, reply(submit()))
    out, _ = await go(client, FakeWorkspace())
    assert out.stop_reason == "submitted"
    assert "was rejected" in client.requests[1][-1].content


async def test_model_error_stops_the_run() -> None:
    out, _ = await go(Script(ConnectionError("refused")), FakeWorkspace())
    assert out.stop_reason == "model_error" and "refused" in (out.error or "")


# --- pieces ----------------------------------------------------------------


def test_render_cell_truncates_stdout_head_and_traceback_tail() -> None:
    c = cell("error", stdout="a" * 60, tb="t" * 55 + "END")
    text = render_cell(c, 50)
    assert "stdout:\n" + "a" * 50 + "\n" + TRUNCATED.format(part="first", limit=50) in text
    assert text.endswith(TRUNCATED.format(part="last", limit=50) + "\n" + "t" * 47 + "END")
    assert "Print a slice, a summary" in text
    assert render_cell(cell(), 50) == "[status: ok, no output]"


def test_reproduction_comparator() -> None:
    assert reproduces(0.1 + 0.2, 0.3) and reproduces(3, 3.0) and reproduces(["a", 1], ["a", 1.0])
    assert not reproduces("3", 3) and not reproduces([1, 2], [2, 1]) and not reproduces(1.0, 1.001)
    assert not reproduces(True, 1)


def test_program_answer_takes_the_last_line_only() -> None:
    assert program_answer('noise\n{"answer": [1, 2]}\n\n') == (True, [1, 2])
    assert program_answer('{"answer": 1}\ntrailing') == (False, None)
    assert program_answer("") == (False, None)


def test_check_submission() -> None:
    ok = {"answer": 1, "files_used": [F], "program": "print()"}
    assert check_submission(ok, [F]) == []
    assert check_submission(ok | {"program": " "}, [F]) == ["program is empty"]
    assert check_submission(ok | {"answer": []}, [F]) == ["answer is an empty list"]
    assert check_submission(ok | {"answer": float("nan")}, [F]) == ["answer is not a finite number"]


def test_prompt_prefix_is_pinned() -> None:
    """The system prompt and tool schemas are the cached prefix. Changing them
    invalidates recorded replays; update this hash deliberately."""
    blob = SYSTEM_PROMPT + json.dumps([t.model_dump() for t in TOOLS], sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == PINNED_PREFIX_SHA256


PINNED_PREFIX_SHA256 = "7a9c3decb8f5f3c9ad8db5d179dfcc74f74c862388763999e31075109c950b52"


def test_package_list_matches_the_image() -> None:
    req = Path(__file__).parents[1] / "ds_research_agent/sandbox/image/requirements.in"
    names = [m[1] for m in re.finditer(r"^([a-z][a-z0-9-]*)", req.read_text(), re.M)]
    assert sorted(names) == sorted(SANDBOX_PACKAGES)
