"""Trusted kernel bridge; runs as root and is PID 1 (kernel sessions).

    bridge.py <session_wall_s> <cpu_s> <max_file_bytes> <max_output_chars> <interrupt_grace_s>

Starts the kernel as uid 1000 under strace and relays the host's requests
(one JSON object per line on stdin) to it. It is the only writer of the
container's stdout: one JSON object per line, each cell's result carrying
the trace lines written since the previous message. Reads, timing, and
outputs come from here; the kernel's own status line is untrusted.

Host requests:  {"op": "exec", "code": str, "timeout_s": int} | {"op": "close"}
Bridge replies: {"type": "ready"} | {"type": "cell", ...} | {"type": "end", ...}
"""

import glob
import json
import os
import select
import signal
import subprocess
import sys
import time

AGENT_UID = 1000
TRACE = (
    "open,openat,openat2,creat,execve,execveat,clone,clone3,fork,vfork,"
    "io_uring_setup,open_by_handle_at"
)

wall_s, cpu_s, fsize, max_chars, grace_s = map(int, sys.argv[1:6])
deadline = time.monotonic() + wall_s
os.makedirs("/audit", mode=0o700, exist_ok=True)
proc = subprocess.Popen(
    ["strace", "-ff", "-qq", "-y", "--seccomp-bpf", "-e", f"trace={TRACE}", "-o", "/audit/t",
     "-u", "agent", "--", "prlimit", f"--cpu={cpu_s}", f"--fsize={fsize}", "--core=0", "--",
     "python3", "/opt/sandbox/kernel.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/audit/strace.err", "wb"),
)  # fmt: skip
assert proc.stdin is not None and proc.stdout is not None
offsets: dict[str, int] = {}
pending = b""


def emit(msg: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def trace_slice() -> dict[str, list[str]]:
    """Complete trace lines written since the last slice, per process."""
    out: dict[str, list[str]] = {}
    for path in sorted(glob.glob("/audit/t.*")):
        start = offsets.get(path, 0)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
        cut = data.rfind(b"\n") + 1
        offsets[path] = start + cut
        lines = data[:cut].decode("utf-8", "backslashreplace").splitlines()
        if lines:
            out[path.rsplit(".", 1)[1]] = lines
    return out


def agent_pids() -> list[int]:
    pids = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if os.stat(f"/proc/{d}").st_uid == AGENT_UID:
                    pids.append(int(d))
            except OSError:
                pass
    return pids


def kernel_pid() -> int | None:
    for pid in agent_pids():
        try:
            with open(f"/proc/{pid}/stat") as f:
                if int(f.read().rsplit(")", 1)[1].split()[1]) == proc.pid:
                    return pid
        except OSError, IndexError, ValueError:
            pass
    return None


def kill_agent() -> None:
    """SIGKILL every uid-1000 process; strace stays up and records it."""
    for pid in agent_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class _Eof:
    """The kernel closed its response pipe (it exited)."""


EOF = _Eof()


def read_response(timeout: float) -> bytes | _Eof | None:
    """One line from the kernel; None on timeout, EOF if the kernel is gone."""
    global pending
    end = time.monotonic() + max(timeout, 0)
    fd = proc.stdout.fileno()  # type: ignore[union-attr]
    while b"\n" not in pending:
        left = end - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            return None
        chunk = os.read(fd, 65536)
        if not chunk:
            return EOF
        pending += chunk
    line, _, pending = pending.partition(b"\n")
    return line


def bounded(path: str) -> tuple[str, bool]:
    try:
        with open(path, "rb") as f:
            data = f.read(max_chars * 4 + 1)
            more = f.read(1) != b""
    except FileNotFoundError:
        return "", False
    text = data.decode("utf-8", "replace")
    return text[:max_chars], more or len(text) > max_chars


emit({"type": "ready", "strace": subprocess.run(
    ["strace", "-V"], capture_output=True, text=True).stdout.splitlines()[0]})  # fmt: skip
cell = 0
alive = True
session_timed_out = False
for raw in sys.stdin:
    req = json.loads(raw)
    if req.get("op") != "exec":
        break
    cell += 1
    if not alive:
        emit({"type": "cell", "cell": cell, "status": "dead", "traceback": None,
              "trace": trace_slice()})  # fmt: skip
        continue
    budget = min(float(req["timeout_s"]), deadline - time.monotonic())
    t0 = time.monotonic_ns()
    proc.stdin.write(json.dumps({"cell": cell, "code": req["code"]}).encode() + b"\n")
    proc.stdin.flush()
    line = read_response(budget)
    status, tb = None, None
    if line is None:
        # Over the cell's limit: interrupt the kernel, then kill it.
        if (pid := kernel_pid()) is not None:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        line = read_response(grace_s)
        if line is None or line is EOF:
            kill_agent()
            alive = False
            status = "timeout"
            session_timed_out = time.monotonic() >= deadline
        else:
            status = "timeout_interrupted"
    if line is EOF:
        kill_agent()
        alive = False
        status = status or "dead"  # the kernel exited (crash, OOM, CPU limit)
    elif isinstance(line, bytes):
        try:
            resp = json.loads(line)
            if resp.get("cell") != cell:
                raise ValueError("cell mismatch")
            status = status or str(resp["status"])
            tb = resp.get("traceback")
        except ValueError, KeyError, TypeError:
            kill_agent()
            alive = False
            status = "protocol_error"
    if alive and time.monotonic() >= deadline:
        kill_agent()  # the session's wall clock is spent
        alive = False
        session_timed_out = True
    stdout, out_trunc = bounded(f"/out/{cell}.stdout")
    stderr, err_trunc = bounded(f"/out/{cell}.stderr")
    if isinstance(tb, str) and len(tb) > max_chars:
        tb = tb[-max_chars:]
    emit({
        "type": "cell", "cell": cell, "status": status, "traceback": tb,
        "stdout": stdout, "stdout_truncated": out_trunc,
        "stderr": stderr, "stderr_truncated": err_trunc,
        "wall_ns": time.monotonic_ns() - t0, "trace": trace_slice(),
    })  # fmt: skip

# Close: let the kernel exit on EOF, then make sure nothing outlives strace.
proc.stdin.close()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    kill_agent()
    proc.wait(timeout=30)
kill_agent()
emit({
    "type": "end", "cells": cell, "strace_exit": proc.returncode,
    "session_timed_out": session_timed_out, "trace": trace_slice(),
})  # fmt: skip
