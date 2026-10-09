"""Parse the trusted wrapper's output into observed data-file access.

The wrapper (``image/entrypoint.sh``) prints ``=== key value`` metadata lines
and one ``=== file /audit/t.<pid>`` section per traced process, written by
``strace -ff -y``. Observed reads are successful opens (and execs) whose
resolved path is under the data mount. The audit fails closed: anything
that could hide a read marks it incomplete, and an incomplete audit cannot
pass provenance verification.
"""

from __future__ import annotations

import codecs
import re

from pydantic import BaseModel, ConfigDict

DATA_ROOT = "/data"

_OPEN_RE = re.compile(
    r"^(open|openat|openat2|creat)\((.*)\) = (-?\d+)(?:<(.*)>)?(?: (\w+) \((.*)\))?$"
)
_OPEN_PREFIX = ("open(", "openat(", "openat2(", "creat(")
_EXEC_RE = re.compile(r'^execve(?:at)?\((?:\d+<[^>]*>, )?"((?:[^"\\]|\\.)*)", .*\) = (-?\d+)')
# Lines that carry no file access: process lifecycle, signals, and the
# remaining traced syscalls (checked separately below).
_LIFECYCLE = ("+++", "---", "execve", "execveat", "clone(", "clone3(", "fork(", "vfork(")
_ESCAPES = ("io_uring_setup(", "open_by_handle_at(")
_RET_RE = re.compile(r"\) = (-?\d+)")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuditResult(_Frozen):
    complete: bool
    # Reasons the audit is incomplete; empty when complete.
    issues: list[str]
    processes: int
    # Container paths under DATA_ROOT, sorted.
    data_reads: list[str]
    data_dirs_listed: list[str]
    # Data opens the kernel refused (absent or unselected files, writes).
    failed_data_opens: list[str]
    # Attempts the sandbox refused that would have escaped the trace.
    blocked_escapes: list[str]
    unparsed_lines: int


class WrapperOutput(_Frozen):
    meta: dict[str, str]
    traces: dict[str, list[str]]


def split_wrapper_output(stdout: str) -> WrapperOutput:
    meta: dict[str, str] = {}
    traces: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stdout.splitlines():
        if line.startswith("=== "):
            key, _, value = line[4:].partition(" ")
            if key == "file":
                current = traces.setdefault(value.rsplit(".", 1)[-1], [])
            else:
                meta[key] = value
                current = None
        elif current is not None:
            current.append(line)
    return WrapperOutput(meta=meta, traces=traces)


def _unescape(path: str) -> str:
    """Undo strace's quoting (octal and \\x escapes, \\", \\\\)."""
    raw = codecs.escape_decode(path.encode("utf-8"))[0]
    assert isinstance(raw, bytes)
    return raw.decode("utf-8", "surrogateescape")


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _ret(line: str) -> int | None:
    m = _RET_RE.search(line)
    return int(m.group(1)) if m else None


def observe(out: WrapperOutput, *, timed_out: bool, data_root: str = DATA_ROOT) -> AuditResult:
    issues: list[str] = []
    reads: set[str] = set()
    dirs: set[str] = set()
    failed: list[str] = []
    blocked: list[str] = []
    unparsed = 0
    if "end" not in out.meta:
        issues.append("wrapper output truncated (no end marker)")
    if not out.traces:
        issues.append("no traced processes")
    if timed_out:
        issues.append("timed out: processes outlived the tracer")
    for pid, lines in out.traces.items():
        for line in lines:
            if line.startswith(_OPEN_PREFIX):
                m = _OPEN_RE.match(line)
                if m is None:
                    issues.append(f"unparsed open in {pid}: {line[:200]}")
                    continue
                _, args, ret, path, _, _ = m.groups()
                if int(ret) >= 0:
                    if not path:
                        issues.append(f"undecorated fd in {pid}: {line[:200]}")
                    elif _under(p := _unescape(path), data_root):
                        (dirs if "O_DIRECTORY" in args else reads).add(p)
                elif data_root in args:
                    failed.append(f"{pid}: {line[:300]}")
                continue
            if line.startswith(("execve(", "execveat(")):
                m = _EXEC_RE.match(line)
                if m and int(m.group(2)) == 0 and _under(p := _unescape(m.group(1)), data_root):
                    reads.add(p)
                continue
            if line.startswith(("clone(", "clone3(")) and "CLONE_UNTRACED" in line:
                ret = _ret(line)
                if ret is not None and ret >= 0:
                    issues.append(f"untraced child in {pid}: {line[:200]}")
                else:
                    blocked.append(f"{pid}: {line[:200]}")
                continue
            if line.startswith(_ESCAPES):
                ret = _ret(line)
                if ret is not None and ret >= 0:
                    issues.append(f"untraceable I/O in {pid}: {line[:200]}")
                else:
                    blocked.append(f"{pid}: {line[:200]}")
                continue
            if not line.startswith(_LIFECYCLE):
                unparsed += 1
    return AuditResult(
        complete=not issues,
        issues=issues,
        processes=len(out.traces),
        data_reads=sorted(reads),
        data_dirs_listed=sorted(dirs),
        failed_data_opens=failed,
        blocked_escapes=blocked,
        unparsed_lines=unparsed,
    )
