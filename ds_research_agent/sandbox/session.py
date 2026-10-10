"""A persistent per-run Python kernel in one restricted container.

Exploration state (loaded data, variables) survives between cells. The
container has the same isolation as ``SandboxRunner.run`` and runs under
``--init`` so orphaned children are reaped. A trusted bridge (PID 2 under
tini, root) relays each cell to the kernel (uid 1000, under strace for its
whole life) and returns the cell's trace slice with its result.

Per-cell reads are attributed by when trace lines were written, so a read
by a background thread or process can land in a later cell's slice. The
session audit, from the merged trace, is the complete record. The verifier
never reuses a session: it reruns the final program with
``SandboxRunner.run`` in a fresh container.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ds_research_agent.sandbox.audit import AuditResult, WrapperOutput, observe
from ds_research_agent.sandbox.runner import (
    InputMount,
    ObservedRead,
    SandboxError,
    SandboxRunner,
    _host_path,
)

CellStatus = Literal[
    # ok / error / interrupted come from the kernel (untrusted); the rest
    # are decided by the bridge.
    "ok", "error", "interrupted", "timeout_interrupted", "timeout", "dead", "protocol_error"
]  # fmt: skip


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CellResult(_Frozen):
    cell: int
    code: str
    status: CellStatus
    traceback: str | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    wall_s: float
    # Reads whose trace lines were written during this cell.
    audit: AuditResult
    observed_reads: list[ObservedRead]

    @property
    def kernel_alive(self) -> bool:
        return self.status not in ("timeout", "dead", "protocol_error")


class SessionEnd(_Frozen):
    container: str
    image_id: str
    cells: int
    session_timed_out: bool
    strace_exit: int | None
    wall_s: float
    # Merged over the whole session, kernel startup included.
    audit: AuditResult
    observed_reads: list[ObservedRead]
    scratch_dir: Path


class KernelSession:
    """Use as a context manager; ``close()`` returns the session audit."""

    def __init__(
        self,
        runner: SandboxRunner,
        inputs: Sequence[InputMount],
        *,
        scratch: Path | None = None,
    ) -> None:
        s = runner.settings
        self._s = s
        self._mounts, self.scratch, self._call = runner._prepare(inputs, scratch)
        self.container = f"dsra-krn-{uuid.uuid4().hex[:12]}"
        self.image_id = runner._image_id()
        cmd = [s.docker, "run", "-i", "--init"]
        cmd += runner._container_args(self.container, self._call, self.scratch, self._mounts)
        cmd += [
            self.image_id, "kernel", str(s.session_timeout_s), str(s.cpu_time_s),
            str(s.max_file_bytes), str(s.max_output_chars), str(s.interrupt_grace_s),
        ]  # fmt: skip
        self._t0 = time.monotonic()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._traces: dict[str, list[str]] = {}
        self._cells = 0
        self._end: SessionEnd | None = None
        ready = self._next(timeout=60)
        if ready.get("type") != "ready":
            raise SandboxError(f"kernel did not start: {ready}")

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _next(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise SandboxError(f"{self.container}: no reply within {timeout:.0f} s") from None
        if line is None:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise SandboxError(f"{self.container}: bridge exited: {err[-2000:]}")
        msg: dict[str, Any] = json.loads(line)
        for pid, lines in msg.get("trace", {}).items():
            self._traces.setdefault(pid, []).extend(lines)
        return msg

    def _kill(self) -> None:
        subprocess.run([self._s.docker, "kill", self.container], capture_output=True, timeout=60)

    def _reads(self, audit: AuditResult) -> list[ObservedRead]:
        return [
            ObservedRead(container_path=p, host_path=_host_path(p, self._mounts))
            for p in audit.data_reads
        ]

    def execute(self, code: str, timeout_s: int | None = None) -> CellResult:
        if self._end is not None:
            raise SandboxError("session is closed")
        timeout = timeout_s or self._s.cell_timeout_s
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps({"op": "exec", "code": code, "timeout_s": timeout}))
        self._proc.stdin.write("\n")
        self._proc.stdin.flush()
        msg = self._next(timeout + self._s.interrupt_grace_s + self._s.host_grace_s)
        self._cells = int(msg["cell"])
        audit = observe(
            WrapperOutput(meta={}, traces=msg.get("trace", {})), timed_out=False, cell_slice=True
        )
        return CellResult(
            cell=self._cells,
            code=code,
            status=msg["status"],
            traceback=msg.get("traceback"),
            stdout=msg.get("stdout", ""),
            stderr=msg.get("stderr", ""),
            stdout_truncated=bool(msg.get("stdout_truncated")),
            stderr_truncated=bool(msg.get("stderr_truncated")),
            wall_s=round(int(msg.get("wall_ns", 0)) / 1e9, 3),
            audit=audit,
            observed_reads=self._reads(audit),
        )

    def close(self) -> SessionEnd:
        if self._end is not None:
            return self._end
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps({"op": "close"}) + "\n")
            self._proc.stdin.close()
            msg = self._next(60 + self._s.host_grace_s)
            ended = msg.get("type") == "end"
        except SandboxError, BrokenPipeError:
            msg, ended = {}, False
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._kill()
        timed_out = bool(msg.get("session_timed_out"))
        # At the deadline the bridge kills the tracees, not strace, so a
        # timed-out session's trace is still complete (unlike program mode).
        audit = observe(
            WrapperOutput(meta={"end": ""} if ended else {}, traces=self._traces),
            timed_out=False,
        )
        self._end = SessionEnd(
            container=self.container,
            image_id=self.image_id,
            cells=self._cells,
            session_timed_out=timed_out,
            strace_exit=msg.get("strace_exit"),
            wall_s=round(time.monotonic() - self._t0, 3),
            audit=audit,
            observed_reads=self._reads(audit),
            scratch_dir=self.scratch,
        )
        return self._end

    def __enter__(self) -> KernelSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._end is None:
            try:
                self.close()
            except SandboxError:
                self._kill()
