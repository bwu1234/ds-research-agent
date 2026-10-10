"""Run one Python program in a fresh, restricted, read-audited container.

Each call starts a new container: no network, read-only root filesystem,
the selected input files bind-mounted read-only under ``/data`` (one mount
per file or directory, so unselected siblings do not exist), a writable
scratch directory, all capabilities dropped except those the trusted
wrapper needs for ``strace -u``, the derived seccomp profile, and
memory/CPU/process limits. The program runs as uid 1000 with no
capabilities; the wrapper, not the program, reports what it read.

The persistent per-run kernel (D3 step c) builds on this; until then each
call is stateless apart from the scratch directory.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from ds_research_agent.config import SandboxSettings
from ds_research_agent.sandbox.audit import (
    DATA_ROOT,
    AuditResult,
    observe,
    split_wrapper_output,
)

IMAGE_DIR = Path(__file__).resolve().parent / "image"
SECCOMP_PROFILE = IMAGE_DIR / "seccomp.json"
PROGRAM_PATH = "/program/main.py"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputMount(_Frozen):
    """A host file or directory exposed read-only at ``container_path``."""

    host_path: Path
    container_path: str


class ObservedRead(_Frozen):
    container_path: str
    # None when the path resolves to no input mount (should not happen:
    # nothing else exists under /data).
    host_path: Path | None


class ImageInfo(_Frozen):
    tag: str
    image_id: str
    packages: list[str]
    strace: str


class SandboxRun(_Frozen):
    container: str
    image_id: str
    exit_code: int | None
    timed_out: bool
    wall_s: float
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    audit: AuditResult
    observed_reads: list[ObservedRead]
    scratch_dir: Path


class SandboxError(RuntimeError):
    """The sandbox itself failed (not the program inside it)."""


def _validate_mounts(inputs: Sequence[InputMount]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    out: list[tuple[Path, str]] = []
    for m in inputs:
        cp = PurePosixPath(m.container_path)
        if not cp.is_absolute() or ".." in cp.parts or cp.parent == cp:
            raise ValueError(f"bad container path: {m.container_path!r}")
        if cp.parts[:2] != ("/", DATA_ROOT.strip("/")) or len(cp.parts) < 3:
            raise ValueError(f"inputs mount under {DATA_ROOT}/: {m.container_path!r}")
        if any(c in str(cp) for c in ",:\n"):
            raise ValueError(f"unsupported character in {m.container_path!r}")
        if str(cp) in seen:
            raise ValueError(f"duplicate container path: {cp}")
        seen.add(str(cp))
        # Docker follows a symlinked source; resolve it here so the recorded
        # host path is the file actually mounted.
        host = m.host_path.resolve(strict=True)
        if any(c in str(host) for c in ",\n"):
            raise ValueError(f"unsupported character in {host}")
        out.append((host, str(cp)))
    return out


def _host_path(container_path: str, mounts: list[tuple[Path, str]]) -> Path | None:
    for host, cp in mounts:
        if container_path == cp:
            return host
        if container_path.startswith(cp + "/"):
            return host / container_path[len(cp) + 1 :]
    return None


def _read_bounded(path: Path, limit: int) -> tuple[str, bool]:
    if not path.exists():
        return "", False
    with path.open("rb") as f:
        # Bytes, not characters, bound the read; decoding may shorten it.
        data = f.read(limit * 4 + 1)
        more = f.read(1) != b""
    text = data.decode("utf-8", "replace")
    truncated = more or len(text) > limit
    return text[:limit], truncated


class SandboxRunner:
    def __init__(self, settings: SandboxSettings) -> None:
        self.settings = settings

    def _docker(self, *args: str, timeout: float = 120) -> str:
        r = subprocess.run(
            [self.settings.docker, *args], capture_output=True, text=True, timeout=timeout
        )
        if r.returncode != 0:
            raise SandboxError(f"docker {args[0]} failed ({r.returncode}): {r.stderr[-2000:]}")
        return r.stdout

    def build_image(self) -> ImageInfo:
        self._docker("build", "-q", "-t", self.settings.image, str(IMAGE_DIR), timeout=1800)
        return self.image_info()

    def image_info(self) -> ImageInfo:
        image_id = self._docker("image", "inspect", "--format", "{{.Id}}", self.settings.image)
        probe = self._docker(
            "run", "--rm", "--network", "none", "--entrypoint", "sh", self.settings.image,
            "-c", "pip freeze --all; echo '==='; strace -V | head -1",
        )  # fmt: skip
        freeze, _, strace = probe.partition("===\n")
        return ImageInfo(
            tag=self.settings.image,
            image_id=image_id.strip(),
            packages=sorted(freeze.split()),
            strace=strace.strip(),
        )

    def new_scratch(self) -> Path:
        """A new, never-reused scratch directory.

        Docker Desktop serves a stale bind mount if a host directory is
        deleted and recreated at the same path, so paths are never reused.
        """
        root = self.settings.work_root / "scratch"
        root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="run.", dir=root))
        scratch.chmod(0o777)
        return scratch

    def _prepare(
        self, inputs: Sequence[InputMount], scratch: Path | None, program: str = ""
    ) -> tuple[list[tuple[Path, str]], Path, Path]:
        """Validate mounts and make the scratch and per-call directories."""
        s = self.settings
        mounts = _validate_mounts(inputs)
        if scratch is None:
            scratch = self.new_scratch()
        scratch = scratch.resolve(strict=True)
        if not scratch.is_relative_to((s.work_root / "scratch").resolve()):
            raise ValueError(f"scratch must be under {s.work_root / 'scratch'}: {scratch}")
        # Program and output live outside scratch, in a per-call directory.
        call = Path(tempfile.mkdtemp(prefix="call.", dir=s.work_root.resolve()))
        (call / "program").mkdir()
        (call / "program/main.py").write_text(program, encoding="utf-8")
        (call / "program/main.py").chmod(0o644)
        (call / "program").chmod(0o755)
        (call / "out").mkdir(mode=0o777)
        (call / "out").chmod(0o777)
        return mounts, scratch, call

    def _image_id(self) -> str:
        return self._docker("image", "inspect", "--format", "{{.Id}}", self.settings.image).strip()

    def _container_args(
        self, name: str, call: Path, scratch: Path, mounts: list[tuple[Path, str]]
    ) -> list[str]:
        s = self.settings
        cmd = [
            "--rm", "--name", name,
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/tmp:rw,size={s.tmp_mb}m",
            "--tmpfs", "/audit:rw,mode=0700,size=256m",
            "--cap-drop", "ALL",
            "--cap-add", "SYS_PTRACE", "--cap-add", "SETUID", "--cap-add", "SETGID",
            # strace (root) needs this to resolve the uid-1000 tracee's
            # /proc/<pid>/fd; without it -y leaves returned fds undecorated.
            "--cap-add", "DAC_READ_SEARCH",
            # Lets the root wrapper and bridge signal the uid-1000 processes
            # (interrupt a cell, kill leftovers); the program has no caps.
            "--cap-add", "KILL",
            "--security-opt", "no-new-privileges",
            "--security-opt", f"seccomp={SECCOMP_PROFILE}",
            "--ipc", "none",
            "--pids-limit", str(s.pids_limit),
            "--memory", f"{s.memory_mb}m", "--memory-swap", f"{s.memory_mb}m",
            "--cpus", str(s.cpus),
            "--mount", f"type=bind,src={call / 'program'},dst=/program,readonly",
            "--mount", f"type=bind,src={call / 'out'},dst=/out",
            "--mount", f"type=bind,src={scratch},dst=/scratch",
        ]  # fmt: skip
        for host, cp in mounts:
            cmd += ["--mount", f"type=bind,src={host},dst={cp},readonly"]
        return cmd

    def run(
        self,
        program: str,
        inputs: Sequence[InputMount],
        *,
        scratch: Path | None = None,
    ) -> SandboxRun:
        """Run one program in a fresh container (also the verifier's rerun)."""
        s = self.settings
        mounts, scratch, call = self._prepare(inputs, scratch, program)
        name = f"dsra-sbx-{uuid.uuid4().hex[:12]}"
        image_id = self._image_id()
        cmd = [s.docker, "run", *self._container_args(name, call, scratch, mounts)]
        cmd += [
            image_id,
            "program",
            str(s.wall_timeout_s),
            str(s.cpu_time_s),
            str(s.max_file_bytes),
        ]

        t0 = time.monotonic()
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=s.wall_timeout_s + s.host_grace_s
            )
        except subprocess.TimeoutExpired:
            subprocess.run([s.docker, "kill", name], capture_output=True, timeout=60)
            raise SandboxError(f"container {name} exceeded the host deadline") from None
        wall = time.monotonic() - t0
        if r.returncode != 0:
            raise SandboxError(f"docker run failed ({r.returncode}): {r.stderr[-2000:]}")

        wrapper = split_wrapper_output(r.stdout)
        elapsed_ns = int(wrapper.meta.get("elapsed_ns", "0"))
        timed_out = elapsed_ns >= s.wall_timeout_s * 1_000_000_000
        audit = observe(wrapper, timed_out=timed_out)
        exit_raw = wrapper.meta.get("exit")
        stdout, out_trunc = _read_bounded(call / "out/stdout", s.max_output_chars)
        stderr, err_trunc = _read_bounded(call / "out/stderr", s.max_output_chars)
        return SandboxRun(
            container=name,
            image_id=image_id,
            exit_code=int(exit_raw) if exit_raw is not None else None,
            timed_out=timed_out,
            wall_s=round(wall, 3),
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
            audit=audit,
            observed_reads=[
                ObservedRead(container_path=p, host_path=_host_path(p, mounts))
                for p in audit.data_reads
            ],
            scratch_dir=scratch,
        )
