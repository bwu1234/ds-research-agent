"""Where the agent's code runs: the interface the loop sees, and the Docker one.

The loop only needs three operations, so it can be tested with a fake and
replayed from the ledger without Docker. ``DockerWorkspace`` keeps one
kernel session per run, starts a new one (same scratch, state lost) after
the kernel dies, and reruns the final program in a fresh container with
empty scratch.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from ds_research_agent.sandbox import (
    CellResult,
    InputMount,
    KernelSession,
    SandboxRun,
    SandboxRunner,
    SessionEnd,
)


class Workspace(Protocol):
    @property
    def session_id(self) -> str | None:
        """The kernel session that ran the most recent cell, if any."""
        ...

    async def execute(self, code: str, timeout_s: int) -> CellResult: ...

    async def rerun(self, program: str) -> SandboxRun: ...

    async def close(self) -> list[SessionEnd]: ...


class DockerWorkspace:
    def __init__(self, runner: SandboxRunner, inputs: Sequence[InputMount]) -> None:
        self._runner = runner
        self._inputs = list(inputs)
        self._scratch = runner.new_scratch()
        self._session: KernelSession | None = None
        self._last_id: str | None = None
        self._ends: list[SessionEnd] = []

    @property
    def session_id(self) -> str | None:
        return self._last_id

    async def execute(self, code: str, timeout_s: int) -> CellResult:
        session = self._session
        if session is None:
            session = await asyncio.to_thread(
                KernelSession, self._runner, self._inputs, scratch=self._scratch
            )
            self._session, self._last_id = session, session.container
        result = await asyncio.to_thread(session.execute, code, timeout_s)
        if not result.kernel_alive:
            await self._end_session()
        return result

    async def _end_session(self) -> None:
        if self._session is not None:
            session, self._session = self._session, None
            self._ends.append(await asyncio.to_thread(session.close))

    async def rerun(self, program: str) -> SandboxRun:
        return await asyncio.to_thread(self._runner.run, program, self._inputs)

    async def close(self) -> list[SessionEnd]:
        await self._end_session()
        return list(self._ends)
