"""Read-audit spike driver (D0): run each case in a fresh container under audit.

    uv run python scripts/spike_read_audit/run.py --docker docker \
        --out data/measurements/read_audit_spike.json

Builds the spike image, writes synthetic inputs (two CSVs copied from the
test fixtures, a generated 5-point GeoPackage, and a generated ~50 MB CSV),
then runs every case in two audit modes (strace with and without
``--seccomp-bpf``), plus unaudited timing runs. Observed data-file reads come
only from the trace that the root wrapper prints on the container's stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMAGE = "dsra-read-audit-spike"
FIXTURES = REPO / "tests/fixtures/lake/environment/water"

A, B, GPKG, BIG = "/data/a.csv", "/data/b.csv", "/data/geo.gpkg", "/data/big.csv"
# Data files each case reads (by any process it starts).
EXPECTED: dict[str, set[str]] = {
    "py_open": {A},
    "pandas_csv": {B},
    "gdal_gpkg": {GPKG},
    "child_cat": {A},
    "child_python": {B},
    "grandchild_shell": {A},
    "symlink_in_scratch": {A},
    "proc_fd_reopen": {B},
    "mmap_read": {A},
    "copy_then_read": {A},
    "orphan_after_exit": {B},
    "list_dir": set(),
    "kill_tracer": {A},
    "clone_untraced": {B},
    "io_uring": set(),
    "open_by_handle": set(),
    "big_csv": {BIG},
    "nothing": set(),
    "tracee_caps": set(),
}
TIMING_CASES = ["nothing", "big_csv", "pandas_csv"]

OPEN_RE = re.compile(
    r"^(open|openat|openat2|creat)\((.*)\) = (-?\d+)(?:<(.*)>)?(?: (\w+) \((.*)\))?$"
)


def fresh_scratch(root: Path, name: str) -> Path:
    """A new, never-reused scratch directory (one per container run).

    Docker Desktop serves a stale bind mount if a host directory is deleted
    and recreated at the same path, so paths are never reused.
    """
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"{name}.", dir=root))
    scratch.chmod(0o777)
    return scratch


def docker_run(docker: str, data: Path, scratch: Path, *args: str) -> tuple[str, float]:
    cmd = [
        docker, "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m",
        "--tmpfs", "/audit:rw,mode=0700,size=256m",
        "--cap-drop", "ALL",
        "--cap-add", "SYS_PTRACE", "--cap-add", "SETUID", "--cap-add", "SETGID",
        "--cap-add", "CHOWN",
        # strace (root) needs this to resolve fds in the uid-1000 tracee's
        # /proc/<pid>/fd; without it -y leaves returned fds undecorated.
        # The tracee itself ends with no effective capabilities.
        "--cap-add", "DAC_READ_SEARCH",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "128",
        "--memory", "2g", "--cpus", "2",
        "-v", f"{data}:/data:ro",
        "-v", f"{scratch}:/scratch",
        IMAGE, *args,
    ]  # fmt: skip
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall = time.monotonic() - t0
    if r.returncode != 0:
        raise RuntimeError(f"docker run failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout, wall


def parse(stdout: str) -> dict[str, Any]:
    """Split the wrapper's output into metadata and per-process traces."""
    meta: dict[str, Any] = {}
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
    if "end" not in meta:
        raise RuntimeError("wrapper output truncated")
    return {"meta": meta, "traces": traces}


def observe(traces: dict[str, list[str]]) -> dict[str, Any]:
    reads: set[str] = set()
    dirs: set[str] = set()
    failed: list[str] = []
    special: list[str] = []
    undecorated: list[str] = []
    unparsed = 0
    for pid, lines in traces.items():
        for line in lines:
            m = OPEN_RE.match(line)
            if m:
                _, args, ret, path, err, _ = m.groups()
                if int(ret) >= 0 and not path:
                    undecorated.append(f"{pid}: {line}")
                elif int(ret) >= 0 and path.startswith("/data"):
                    (dirs if "O_DIRECTORY" in args else reads).add(path)
                elif int(ret) < 0 and "/data" in args:
                    failed.append(f"{pid}: {line}")
                continue
            if line.startswith(("io_uring_setup", "open_by_handle_at", "clone(", "clone3(")):
                if line.startswith(("io_uring", "open_by")) or "CLONE_UNTRACED" in line:
                    special.append(f"{pid}: {line}")
            elif not line.startswith(("execve", "+++", "---", "vfork", "fork", "clone")):
                unparsed += 1
    return {
        "processes": len(traces),
        "data_reads": sorted(reads),
        "data_dirs_listed": sorted(dirs),
        "failed_data_opens": failed,
        "special_syscalls": special,
        "undecorated_opens": undecorated,
        "unparsed_lines": unparsed,
    }


def prepare(docker: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    data = Path(tempfile.mkdtemp(prefix="data.", dir=work))
    data.chmod(0o755)
    shutil.copyfile(FIXTURES / "station_readings.csv", data / "a.csv")
    shutil.copyfile(REPO / "tests/fixtures/lake/legal/State MSA/Utah 2023.csv", data / "b.csv")
    with (data / "big.csv").open("w") as f:
        f.write("id,x,y,label\n")
        for i in range(1_200_000):
            f.write(f"{i},{i * 0.5:.3f},{(i * 7919) % 10007},row{i % 97}\n")
    scratch = fresh_scratch(work / "scratch", "setup")
    docker_run(docker, data, scratch, "plain", "make_gpkg")
    shutil.copyfile(scratch / "geo.gpkg", data / "geo.gpkg")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docker", default="docker")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=REPO / "data/spike_read_audit")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats")
    args = ap.parse_args()
    d = args.docker

    t0 = time.monotonic()
    subprocess.run([d, "build", "-q", "-t", IMAGE, str(HERE)], check=True, capture_output=True)
    build_s = time.monotonic() - t0
    freeze = subprocess.run(
        [d, "run", "--rm", "--entrypoint", "pip", IMAGE, "freeze"],
        check=True, capture_output=True, text=True,
    ).stdout.split()  # fmt: skip
    info = json.loads(subprocess.run(
        [d, "info", "--format", "{{json .}}"], check=True, capture_output=True, text=True
    ).stdout)  # fmt: skip
    data = prepare(d, args.work)

    result: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "docker": {
            "server": info.get("ServerVersion"),
            "kernel": info.get("KernelVersion"),
            "os": info.get("OperatingSystem"),
            "security": info.get("SecurityOptions"),
        },
        "image_build_s": round(build_s, 1),
        "packages": freeze,
        "cases": {},
        "timing": {},
    }
    for case, expected in EXPECTED.items():
        for mode in ("trace", "trace_ptrace_only"):
            scratch = fresh_scratch(args.work / "scratch", f"{case}.{mode}")
            out, wall = docker_run(d, data, scratch, mode, case)
            parsed = parse(out)
            obs = observe(parsed["traces"])
            claim = (scratch / "out/stdout").read_text().strip()
            stderr = (scratch / "out/stderr").read_text().strip()
            seen = set(obs["data_reads"])
            rec = {
                "expected": sorted(expected),
                "observed": obs,
                "missed": sorted(expected - seen),
                "unexpected": sorted(seen - expected),
                "complete": expected <= seen,
                "exit": parsed["meta"].get("exit"),
                "program_claim": claim[-500:],
                "program_stderr": stderr[-800:],
                "container_wall_s": round(wall, 2),
            }
            result["cases"].setdefault(case, {})[mode] = rec
            result["strace"] = parsed["meta"].get("strace")
            print(
                f"{case:20s} {mode:18s} complete={rec['complete']!s:5s} "
                f"missed={rec['missed']} procs={obs['processes']} exit={rec['exit']} "
                f"claim={claim[-120:]!r}",
                flush=True,
            )
    for case in TIMING_CASES:
        for mode in ("plain", "trace", "trace_ptrace_only"):
            runs = []
            for _ in range(args.repeats):
                scratch = fresh_scratch(args.work / "scratch", f"timing.{case}.{mode}")
                out, wall = docker_run(d, data, scratch, mode, case)
                meta = parse(out)["meta"]
                runs.append({
                    "container_wall_s": round(wall, 3),
                    "inner_s": int(meta["elapsed_ns"]) / 1e9,
                    "claim": (scratch / "out/stdout").read_text().strip(),
                })  # fmt: skip
            result["timing"].setdefault(case, {})[mode] = runs
            inner = sorted(r["inner_s"] for r in runs)
            print(f"timing {case:12s} {mode:18s} inner median {inner[len(inner) // 2]:.2f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
