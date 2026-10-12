"""Replay of agent runs stopped by the wall-clock limit, offline.

A replay takes seconds, so the limit cannot fire in it; ``replay`` takes it
from the recording instead. Each case records a run that a short limit
stops, at a request cut off in flight, at a cell (the comparator's is cut
off; the given-files loop checks the clock after it), then replays it
through ``run.cmd_replay`` with the configured limit.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.ledger import Ledger
from ds_research_agent.models import ChatMessage, ChatResult, ModelClient, ToolSpec
from ds_research_agent.sandbox import CellResult, InputMount
from eval.kramabench import agent_runs, codeagent_runs, run
from eval.kramabench.baselines import Harness, ReplayClient, ReplayMismatch, run_batch
from eval.kramabench.tasks import Task
from tests.test_agent_loop import FakeWorkspace, rerun
from tests.test_codeagent import CodeAgentScript, FakeKernel
from tests.test_given_files import AgentScript, _harness

WALL = 0.75  # seconds: the recorded runs' limit
SLOW = 2.0  # a request or cell this slow outlasts it


class SlowAt:
    """Delegates to ``inner``; request number ``at`` (from 0) outlasts the limit."""

    def __init__(self, inner: ModelClient, at: int) -> None:
        self.inner, self.at, self.n = inner, at, 0

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        self.n += 1
        if self.n - 1 == self.at:
            await asyncio.sleep(SLOW)
        return await self.inner.chat(messages, tools)


class SlowCells(FakeWorkspace):
    async def execute(self, code: str, timeout_s: int) -> CellResult:
        await asyncio.sleep(SLOW)
        return await super().execute(code, timeout_s)


class SlowKernel(FakeKernel):
    async def execute(self, code: str, timeout_s: int) -> CellResult:
        if code != codeagent_runs.SETUP:
            await asyncio.sleep(SLOW)
        return await super().execute(code, timeout_s)


def _record(h: Harness, condition: str, cut: str) -> str:
    agent = h.settings.agent.model_copy(update={"max_wall_s": WALL})
    short = Harness(
        h.settings.model_copy(update={"agent": agent}), h.model, h.ledger, h.split, h.tasks
    )
    keys = h.split.dev[:1]
    t0 = h.tasks[keys[0]]
    answers = {t0.query: t0.gold.answer}
    script = AgentScript(answers) if condition == "given_files" else CodeAgentScript(answers)
    client: ModelClient = SlowAt(script, at=1) if cut == "request" else script
    batch = run.new_batch(
        short.settings, h.model, h.split, condition=condition, task_set="dev", keys=keys,
        repeats=1, batch_id=f"wall-{condition}-{cut}",
    )  # fmt: skip
    h.ledger.add_batch(batch)

    def ws_for(_t: Task, _r: int, mounts: Sequence[InputMount]) -> FakeWorkspace:
        paths = [m.container_path for m in mounts]
        if condition == "codeagent":
            return SlowKernel(paths) if cut == "cell" else FakeKernel(paths)
        ws = SlowCells() if cut == "cell" else FakeWorkspace()
        ws.final = rerun("", reads=paths)
        return ws

    async def task(hh: Harness, b: Any, t: Task, r: int, c: Any) -> Any:
        return await run.AGENT_TASKS[condition][0](hh, b, t, r, c, ws_for, "sha256:img")

    asyncio.run(run_batch(short, batch, lambda _t, _r: client, agent_task=task))
    return batch.batch_id


@pytest.mark.parametrize("condition", ["given_files", "codeagent"])
@pytest.mark.parametrize("cut", ["request", "cell"])
def test_runs_stopped_by_the_wall_limit_replay(tmp_path: Path, condition: str, cut: str) -> None:
    h = _harness(tmp_path)
    bid = _record(h, condition, cut)
    orig = h.ledger.runs(bid)[0]
    assert orig.stop_reason == "max_wall"
    steps = h.ledger.steps(orig.run_id)
    if cut == "request":  # the cut-off request is a step with neither response nor error
        assert (steps[-1].response, steps[-1].error) == (None, None)

    assert run.cmd_replay(argparse.Namespace(batch=[bid]), h.settings) == 0
    (replay,) = [b for b in h.ledger.batches() if b.replay_of == bid]
    again = h.ledger.runs(replay.batch_id)[0]
    assert (again.stop_reason, again.error) == (orig.stop_reason, orig.error)
    replayed = h.ledger.steps(again.run_id)
    assert [s.request_sha256 for s in replayed[: len(steps)]] == [s.request_sha256 for s in steps]
    # Where the live loop checked the clock between steps, the replay sends one
    # more request, answered by the deadline (no request reached the model).
    extra = replayed[len(steps) :]
    assert len(extra) == int((condition, cut) == ("given_files", "cell"))
    assert all("wall-clock limit" in (s.error or "") for s in extra)

    # The default clients stay strict: only a flagged recording replays the limit.
    with pytest.raises(ReplayMismatch):
        asyncio.run(ReplayClient(h.model, []).chat([]))
    ws = agent_runs.ReplayWorkspace([])
    with pytest.raises(ReplayMismatch):
        asyncio.run(ws.execute("print(1)", 5))


@pytest.mark.parametrize("condition", ["given_files", "codeagent"])
@pytest.mark.parametrize("cut", ["steps", "programs"])
def test_runs_not_stopped_by_the_limit_stay_strict_through_cmd_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str, cut: str
) -> None:
    """``cmd_replay`` must not hand the deadline to ordinary runs: past the end of
    their recording they are a mismatch, not a ``max_wall`` stop that would let a
    truncated or edited recording pass as a faithful replay."""
    h = _harness(tmp_path)
    bid = _record(h, condition, "none")
    orig = h.ledger.runs(bid)[0]
    assert orig.stop_reason == "submitted"

    real_steps, real_programs = Ledger.steps, Ledger.programs

    def steps(self: Ledger, run_id: str) -> Any:
        rows = real_steps(self, run_id)
        return rows[:-1] if cut == "steps" and run_id == orig.run_id else rows

    def programs(self: Ledger, run_id: str) -> Any:
        rows = real_programs(self, run_id)
        return [] if cut == "programs" and run_id == orig.run_id else rows

    monkeypatch.setattr(Ledger, "steps", steps)
    monkeypatch.setattr(Ledger, "programs", programs)
    with pytest.raises(ReplayMismatch):
        run.cmd_replay(argparse.Namespace(batch=[bid]), h.settings)
