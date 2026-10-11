"""The D3 minimal comparator (smolagents CodeAgent) in the harness, offline.

A scripted model answers in CodeAgent's text format and a fake kernel
stands in for the sandbox, so the adapter's seams are checked end to end:
code blocks become kernel cells, ``final_answer`` comes back through the
traceback, every model request is a ledger step, the verifier runs, and
the recorded runs replay with no model and no sandbox.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.models import ChatMessage, ChatResult, ToolSpec, Usage
from ds_research_agent.sandbox import CellResult, InputMount, SandboxRun
from eval.kramabench import agent_runs, codeagent_runs, report
from eval.kramabench.baselines import Harness, ReplayClient, run_batch
from eval.kramabench.run import new_batch
from eval.kramabench.scoring import PROFILE
from eval.kramabench.tasks import Task
from tests.test_agent_loop import FakeWorkspace, cell, rerun
from tests.test_given_files import _harness

WRONG_FILES = "wrong-files"  # submits an unknown file first, then the right ones
CRASH = "crash"  # the first cell kills the kernel


def _text(content: str) -> ChatResult:
    return ChatResult(
        message=ChatMessage(role="assistant", content=content),
        usage=Usage(prompt_eval_tokens=100, output_tokens=20, eval_ns=10**9, prompt_eval_ns=10**9),
        done_reason="stop",
        wall_s=2.0,
    )


def _block(code: str) -> ChatResult:
    # Ollama stops at "</code>" (a stop sequence), so the reply has no closing tag.
    return _text(f"Thought: next step.\n<code>\n{code}\n")


def _final(answer: Any, files: Sequence[str]) -> str:
    program = f"print({json.dumps(json.dumps({'answer': answer}))})"
    payload = {"answer": answer, "files_used": list(files), "program": program}
    return f"final_answer(**{json.dumps(payload)})"


class CodeAgentScript:
    """Explores once, then calls final_answer with ``answers[question]``.

    Questions mapped to None never answer (the run hits the step budget)."""

    def __init__(self, answers: dict[str, Any], modes: dict[str, str] | None = None) -> None:
        self.answers, self.modes = answers, modes or {}
        self.requests: list[Sequence[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        assert not tools  # CodeAgent writes code, not tool calls
        self.requests.append(messages)
        task = messages[1].content
        question = re.search(r"^Question: (.*)$", task, re.M)
        assert question is not None
        files = re.findall(r"^- (\S+) \(", task, re.M)
        answer = self.answers[question.group(1)]
        mode = self.modes.get(question.group(1))
        turn = sum(m.role == "assistant" for m in messages)
        if turn == 0 or answer is None:
            return _block("print(1)")
        if mode == WRONG_FILES and turn == 1:
            return _block(_final(answer, ["/data/not-given.csv"]))
        return _block(_final(answer, files))


class FakeKernel(FakeWorkspace):
    """Runs nothing: ok for ordinary cells, and for ``final_answer(**{...})``
    the traceback the sandbox kernel would return. ``crash_first`` kills the
    kernel on the first model cell, so the next session needs setup again."""

    def __init__(self, reads: Sequence[str], crash_first: bool = False) -> None:
        super().__init__()
        self.reads, self.crash = list(reads), crash_first
        self.session = 1

    @property
    def session_id(self) -> str | None:
        return f"dsra-krn-{self.session}"

    async def execute(self, code: str, timeout_s: int) -> CellResult:
        self.executed.append((code, timeout_s))
        if code == codeagent_runs.SETUP:
            return cell()
        if code.startswith("final_answer(**"):
            payload = code.removeprefix("final_answer(**").removesuffix(")")
            tb = (
                'Traceback (most recent call last):\n  File "<cell 3>", line 1\n'
                f"_FinalAnswer: {codeagent_runs.MARKER}{json.dumps(json.loads(payload))}\n"
            )
            return cell("error", tb=tb)
        if self.crash:
            self.crash, self.session = False, self.session + 1
            return cell("dead")
        return cell(stdout="1\n")

    async def rerun(self, program: str) -> SandboxRun:
        self.reruns.append(program)
        return rerun(json.loads(program.removeprefix("print(").removesuffix(")")), self.reads)


def _run(  # type: ignore[no-untyped-def]
    h: Harness,
    keys: tuple[str, ...],
    script: CodeAgentScript,
    reads: dict[str, bool] | None = None,
    crash: Sequence[str] = (),
    batch_id: str = "ca-test",
):
    batch = new_batch(
        h.settings, h.model, h.split, condition="codeagent", task_set="dev", keys=keys,
        repeats=1, batch_id=batch_id,
    )  # fmt: skip
    h.ledger.add_batch(batch)
    kernels: dict[str, FakeKernel] = {}

    def ws_for(t: Task, _r: int, mounts: Sequence[InputMount]) -> FakeKernel:
        paths = [m.container_path for m in mounts] if (reads or {}).get(t.key, True) else []
        kernels[t.key] = FakeKernel(paths, crash_first=t.key in crash)
        return kernels[t.key]

    async def task(hh: Harness, b: Any, t: Task, r: int, c: Any) -> Any:
        return await codeagent_runs.run_codeagent_task(hh, b, t, r, c, ws_for, "sha256:img")

    asyncio.run(run_batch(h, batch, lambda _t, _r: script, agent_task=task))
    return batch, kernels


def test_codeagent_runs_in_the_kernel_records_verifies_and_replays(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    keys = h.split.dev[:4]
    t = [h.tasks[k] for k in keys]
    answers = {
        t[0].query: t[0].gold.answer,  # right, verified
        t[1].query: "wrong",  # reproduces, not strict
        t[2].query: None,  # never answers
        t[3].query: t[3].gold.answer,  # right after a rejected submission
    }
    script = CodeAgentScript(answers, {t[3].query: WRONG_FILES})
    batch, kernels = _run(h, keys, script)
    runs = {r.task_key: r for r in h.ledger.runs(batch.batch_id)}
    stops = [runs[k].stop_reason for k in keys]
    assert stops == ["submitted", "submitted", "max_steps", "submitted"]
    assert batch.system_prompt_sha256 == codeagent_runs.prefix_sha256()

    r0 = runs[keys[0]].run_id
    assert [p.kind for p in h.ledger.programs(r0)] == ["setup", "cell", "cell", "final_rerun"]
    steps = h.ledger.steps(r0)
    assert len(steps) == 2 and steps[0].request["tools"] == []
    first = steps[0].request["messages"]
    assert [m["role"] for m in first] == ["system", "user"]
    assert "final_answer(answer: any, files_used: array, program: string" in first[0]["content"]
    assert codeagent_runs.INSTRUCTIONS in first[0]["content"]
    assert "You have" not in first[1]["content"]  # no turn-budget guidance
    # Append-only: the second request extends the first.
    assert steps[1].request["messages"][:2] == first
    assert kernels[keys[0]].executed[0][0] == codeagent_runs.SETUP

    # The rejected submission came back as an error and cost a step.
    r3 = runs[keys[3]].run_id
    assert len(h.ledger.steps(r3)) == 3
    err = h.ledger.steps(r3)[2].request["messages"][-1]["content"]
    assert "files_used lists files not given" in err and "Nothing was submitted" in err
    assert kernels[keys[3]].reruns == [h.ledger.submission(r3).program]  # type: ignore[union-attr]

    assert len(h.ledger.steps(runs[keys[2]].run_id)) == h.settings.agent.max_steps
    verified = {k: h.ledger.score(runs[k].run_id, PROFILE) for k in keys}
    assert [(s.strict, s.verified_success) for s in verified.values() if s] == [
        (True, True), (False, False), (False, False), (True, True),
    ]  # fmt: skip
    out = report.summarise(h.ledger, batch, h.tasks, 200, 0)
    assert out["agent"]["submitted"] == 3 and out["agent"]["access_verified"] == 3

    # Replay: recorded responses and programs, no model, no sandbox.
    replay = new_batch(
        h.settings, h.model, h.split, condition="codeagent", task_set="dev", keys=keys,
        repeats=1, batch_id="ca-replay", replay_of=batch,
    )  # fmt: skip
    h.ledger.add_batch(replay)
    steps_of = {k: h.ledger.steps(runs[k].run_id) for k in keys}
    progs_of = {k: h.ledger.programs(runs[k].run_id) for k in keys}

    async def task(hh: Harness, b: Any, tk: Task, r: int, c: Any) -> Any:
        ws = agent_runs.ReplayWorkspace(progs_of[tk.key])
        return await codeagent_runs.run_codeagent_task(hh, b, tk, r, c, lambda *_: ws, "sha256:img")

    asyncio.run(
        run_batch(
            h, replay, lambda tk, _r: ReplayClient(h.model, steps_of[tk.key]), agent_task=task
        )
    )
    for a, b in zip(h.ledger.runs(batch.batch_id), h.ledger.runs("ca-replay"), strict=True):
        assert (a.stop_reason, a.error) == (b.stop_reason, b.error)
        sa, sb = h.ledger.score(a.run_id, PROFILE), h.ledger.score(b.run_id, PROFILE)
        assert sa and sb and (sa.score, sa.strict, sa.verified_success) == (
            sb.score, sb.strict, sb.verified_success,
        )  # fmt: skip
        va, vb = h.ledger.verification(a.run_id), h.ledger.verification(b.run_id)
        assert (va and va.model_dump(exclude={"run_id"})) == (
            vb and vb.model_dump(exclude={"run_id"})
        )


def test_kernel_restart_reruns_setup_and_reports_the_restart(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    keys = h.split.dev[:1]
    task = h.tasks[keys[0]]
    script = CodeAgentScript({task.query: task.gold.answer})
    batch, kernels = _run(h, keys, script, crash=keys)
    run = h.ledger.runs(batch.batch_id)[0]
    assert run.stop_reason == "submitted"
    progs = h.ledger.programs(run.run_id)
    assert [p.kind for p in progs] == ["setup", "cell", "setup", "cell", "final_rerun"]
    assert progs[0].session != progs[2].session
    second = h.ledger.steps(run.run_id)[1].request["messages"][-1]["content"]
    assert "kernel was restarted (dead)" in second


def test_sandbox_failure_is_a_stop_reason(tmp_path: Path) -> None:
    from ds_research_agent.sandbox import SandboxError

    h = _harness(tmp_path)
    keys = h.split.dev[:1]
    task = h.tasks[keys[0]]
    batch = new_batch(
        h.settings, h.model, h.split, condition="codeagent", task_set="dev", keys=keys,
        repeats=1, batch_id="ca-sbx",
    )  # fmt: skip
    h.ledger.add_batch(batch)

    class Broken(FakeKernel):
        async def execute(self, code: str, timeout_s: int) -> CellResult:
            raise SandboxError("container vanished")

    async def run(hh: Harness, b: Any, t: Task, r: int, c: Any) -> Any:
        return await codeagent_runs.run_codeagent_task(hh, b, t, r, c, lambda *_: Broken([]), None)

    script = CodeAgentScript({task.query: task.gold.answer})
    asyncio.run(run_batch(h, batch, lambda _t, _r: script, agent_task=run))
    r = h.ledger.runs(batch.batch_id)[0]
    assert (r.stop_reason, r.error) == ("sandbox_error", "container vanished")
    assert len(h.ledger.steps(r.run_id)) == 1


def test_harness_errors_are_raised_not_shown_to_the_model(tmp_path: Path) -> None:
    """CodeAgent turns executor exceptions into feedback for the model; a harness
    bug must instead end the batch (it once surfaced as a SQLite thread error)."""
    h = _harness(tmp_path)
    keys = h.split.dev[:1]
    task = h.tasks[keys[0]]
    batch = new_batch(
        h.settings, h.model, h.split, condition="codeagent", task_set="dev", keys=keys,
        repeats=1, batch_id="ca-bug",
    )  # fmt: skip
    h.ledger.add_batch(batch)

    class Buggy(FakeKernel):
        @property
        def session_id(self) -> str | None:
            raise KeyError("harness bug")

    async def run(hh: Harness, b: Any, t: Task, r: int, c: Any) -> Any:
        return await codeagent_runs.run_codeagent_task(hh, b, t, r, c, lambda *_: Buggy([]), None)

    script = CodeAgentScript({task.query: task.gold.answer})
    with pytest.raises(KeyError, match="harness bug"):
        asyncio.run(run_batch(h, batch, lambda _t, _r: script, agent_task=run))
    assert len(script.requests) == 1


def test_prefix_is_pinned() -> None:
    """CodeAgent's rendered system prompt (smolagents 1.26.0) and stop options.
    A smolagents upgrade or prompt change must re-pin this deliberately."""
    assert codeagent_runs.prefix_sha256() == (
        "427ce7f5af89523673245b5980d9306783ebc1ebb94b9baa3c258bd1a8801132"
    )


@pytest.mark.parametrize("bad", [{"answer": 1}, "just text"])
def test_submission_check_rejects_partial_final_answers(bad: Any) -> None:
    check = codeagent_runs._submission_check(["/data/a.csv"])
    with pytest.raises(ValueError):
        check(bad, None)
