"""Sandbox isolation and read-audit checks against real containers.

Deterministic and model-free, but they need a running Docker daemon, so they
carry the ``docker`` marker and are excluded by default:

    uv run pytest -m docker
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from ds_research_agent.sandbox import InputMount, SandboxRun, SandboxRunner

pytestmark = pytest.mark.docker

LAKE = Path(__file__).resolve().parents[1] / "fixtures/lake"
WATER = LAKE / "environment/water"
A = "/data/water/readings.csv"

Run = Callable[..., SandboxRun]


@pytest.fixture
def run(runner: SandboxRunner) -> Run:
    """Run a program with readings.csv mounted at A (and nothing else)."""

    def _run(program: str, inputs: list[InputMount] | None = None, **kw: object) -> SandboxRun:
        if inputs is None:
            inputs = [InputMount(host_path=WATER / "station_readings.csv", container_path=A)]
        return runner.run(textwrap.dedent(program), inputs, **kw)  # type: ignore[arg-type]

    return _run


def result(run: SandboxRun) -> dict[str, object]:
    """The program reports through its last stdout line as JSON."""
    out: dict[str, object] = json.loads(run.stdout.strip().splitlines()[-1])
    return out


# --- isolation -----------------------------------------------------------


def test_no_network(run: Run) -> None:
    r = run("""
        import json, socket
        out = {}
        for name, fn in {
            "tcp": lambda: socket.create_connection(("1.1.1.1", 53), timeout=2),
            "dns": lambda: socket.getaddrinfo("example.com", 80),
        }.items():
            try:
                fn(); out[name] = "connected"
            except OSError as e:
                out[name] = type(e).__name__
        # Tunnel devices exist (down) in every namespace; routes are the test.
        out["routes"] = len(open("/proc/net/route").read().splitlines()) - 1
        print(json.dumps(out))
    """)
    out = result(r)
    assert out["tcp"] != "connected" and out["dns"] != "connected"
    assert out["routes"] == 0


def test_only_selected_inputs_exist(run: Run) -> None:
    r = run(f"""
        import json, os
        out = {{"data": sorted(os.listdir("/data")), "water": sorted(os.listdir("/data/water"))}}
        for p in ["/data/water/stations.json", "{WATER / "stations.json"}", "{LAKE}", "/Users",
                  "/root/.ssh", "/var/run/docker.sock"]:
            out[p] = os.path.exists(p)
        print(json.dumps(out))
    """)
    out = result(r)
    assert out.pop("data") == ["water"] and out.pop("water") == ["readings.csv"]
    assert not any(out.values()), out


def test_writes_only_to_scratch_and_tmp(run: Run) -> None:
    before = (WATER / "station_readings.csv").read_bytes()
    r = run("""
        import json
        out = {}
        for p in [A := "/data/water/readings.csv", "/data/new.txt", "/etc/x", "/x",
                  "/program/main.py", "/usr/lib/x", "/scratch/ok.txt", "/tmp/ok.txt"]:
            try:
                with open(p, "a") as f: f.write("x")
                out[p] = "written"
            except OSError as e:
                out[p] = e.strerror
        print(json.dumps(out))
    """)
    out = result(r)
    written = {p for p, v in out.items() if v == "written"}
    assert written == {"/scratch/ok.txt", "/tmp/ok.txt"}, out
    assert (r.scratch_dir / "ok.txt").read_text() == "x"
    assert (WATER / "station_readings.csv").read_bytes() == before


def test_scratch_persists_when_reused(runner: SandboxRunner, run: Run) -> None:
    scratch = runner.new_scratch()
    run("open('/scratch/state.txt', 'w').write('kept')", scratch=scratch)
    r = run("print(open('/scratch/state.txt').read())", scratch=scratch)
    assert r.stdout.strip() == "kept"


def test_no_host_environment_or_capabilities(run: Run) -> None:
    r = run("""
        import json, os
        caps = {l.split(":")[0]: l.split()[1] for l in open("/proc/self/status")
                if l.startswith(("Cap", "NoNewPrivs"))}
        print(json.dumps({"uid": os.getuid(), "env": sorted(os.environ), **caps}))
    """)
    out = result(r)
    assert out["uid"] == 1000
    # The bounding set keeps the container's caps, but with nothing permitted,
    # inheritable, or ambient, and no_new_privs set, exec cannot regain them.
    zero = "0000000000000000"
    assert out["CapEff"] == out["CapPrm"] == out["CapInh"] == out["CapAmb"] == zero
    assert out["NoNewPrivs"] == "1"
    env = set(out["env"])  # type: ignore[arg-type]
    assert env <= {
        "HOME", "HOSTNAME", "LANG", "MPLCONFIGDIR", "PATH", "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED", "PWD", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256",
    }, env  # fmt: skip


# --- resource limits -----------------------------------------------------


def test_wall_timeout_kills_and_marks_audit_incomplete(run: Run) -> None:
    r = run("import time\nwhile True: time.sleep(1)")
    assert r.timed_out
    assert not r.audit.complete
    assert r.wall_s < 8 + 20


def test_cpu_limit(run: Run) -> None:
    r = run("while True: pass")
    assert not r.timed_out
    assert r.exit_code != 0  # SIGXCPU, then SIGKILL at the hard limit


def test_memory_limit(run: Run) -> None:
    r = run("""
        chunks = []
        while True: chunks.append(bytearray(64 * 1024 * 1024))
    """)
    assert r.exit_code != 0
    assert not r.timed_out


def test_process_limit(run: Run) -> None:
    r = run("""
        import json, os, time
        n = 0
        try:
            while n < 1000:
                if os.fork() == 0:
                    time.sleep(30); os._exit(0)
                n += 1
        except OSError as e:
            err = e.strerror
        print(json.dumps({"forked": n, "error": err}), flush=True)
        os.killpg(0, 9)
    """)
    out = result(r)
    assert out["forked"] < 64  # type: ignore[operator]


def test_output_is_bounded(run: Run) -> None:
    r = run("import sys\nfor _ in range(10_000): sys.stdout.write('y' * 1000 + '\\n')")
    assert r.stdout_truncated
    assert len(r.stdout) == 2_000


# --- read audit ----------------------------------------------------------


def test_reads_observed_with_host_paths(run: Run) -> None:
    r = run("""
        import pandas as pd, subprocess
        pd.read_csv("/data/water/readings.csv")
        subprocess.run(["sh", "-c", "sh -c 'cat /data/water/readings.csv' >/dev/null"])
    """)
    assert r.exit_code == 0 and r.audit.complete
    assert [o.container_path for o in r.observed_reads] == [A]
    assert r.observed_reads[0].host_path == (WATER / "station_readings.csv").resolve()


def test_no_reads_observed_when_none_made(run: Run) -> None:
    r = run("print('hi')")
    assert r.audit.complete and r.observed_reads == []


def test_native_reader_observed(runner: SandboxRunner, run: Run) -> None:
    # Write a GeoPackage in one run, mount it as input to the next.
    scratch = runner.new_scratch()
    made = run(
        """
        import struct, numpy as np, pyogrio.raw
        g = np.array([struct.pack("<BIdd", 1, 1, i / 10, i / 20) for i in range(5)], dtype=object)
        pyogrio.raw.write("/scratch/geo.gpkg", g, [np.array(list("abcde"), dtype=object)],
                          fields=["name"], geometry_type="Point", crs="EPSG:4326", driver="GPKG")
        """,
        scratch=scratch,
    )
    assert made.exit_code == 0, made.stderr
    r = run(
        "import pyogrio.raw\nprint(len(pyogrio.raw.read('/data/geo/sites.gpkg')[2]))",
        [InputMount(host_path=scratch / "geo.gpkg", container_path="/data/geo/sites.gpkg")],
    )
    assert r.stdout.strip() == "5" and r.audit.complete
    assert "/data/geo/sites.gpkg" in r.audit.data_reads


def test_untraced_clone_is_refused(run: Run) -> None:
    r = run("""
        import ctypes, json, os, platform
        libc = ctypes.CDLL(None, use_errno=True)
        nr = {"aarch64": 220, "x86_64": 56}[platform.machine()]
        flags = ctypes.c_long(0x00800000 | 17)  # CLONE_UNTRACED | SIGCHLD
        pid = libc.syscall(ctypes.c_long(nr), flags, *[ctypes.c_long(0)] * 4)
        if pid == 0:
            open("/data/water/readings.csv").read(); os._exit(0)
        if pid > 0:
            os.waitpid(pid, 0)
        print(json.dumps({"pid": pid, "errno": ctypes.get_errno()}))
    """)
    out = result(r)
    assert out["pid"] == -1 and out["errno"] == 1  # EPERM from the seccomp profile
    # Docker's ERRNO outranks strace's TRACE filter, so the refused call
    # never reaches the tracer; the refusal itself is the guarantee.
    assert r.audit.complete
    assert r.observed_reads == []


def test_clone3_unavailable_and_threads_still_work(run: Run) -> None:
    r = run("""
        import concurrent.futures as cf, ctypes, json, subprocess
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.syscall(ctypes.c_long(435), ctypes.c_long(0), ctypes.c_long(0))
        with cf.ThreadPoolExecutor(4) as ex:
            total = sum(ex.map(lambda i: i, range(10)))
        child = subprocess.run(["true"]).returncode
        print(json.dumps({"rc": rc, "errno": ctypes.get_errno(), "total": total, "child": child}))
    """)
    out = result(r)
    assert out == {"rc": -1, "errno": 38, "total": 45, "child": 0}  # ENOSYS


def test_program_cannot_kill_tracer(run: Run) -> None:
    r = run("""
        import json, os, signal
        try:
            os.kill(os.getppid(), signal.SIGKILL); out = "killed"
        except PermissionError:
            out = "refused"
        open("/data/water/readings.csv").read()
        print(json.dumps({"kill": out}))
    """)
    assert result(r)["kill"] == "refused"
    assert r.audit.complete and r.audit.data_reads == [A]


def test_detached_grandchild_read_observed(run: Run) -> None:
    r = run("""
        import subprocess
        subprocess.Popen(["sh", "-c", "sleep 1; cat /data/water/readings.csv >/dev/null"],
                         start_new_session=True)
    """)
    assert r.audit.complete and r.audit.data_reads == [A]


def test_mount_validation(runner: SandboxRunner) -> None:
    src = WATER / "station_readings.csv"
    for bad in ["/etc/passwd", "/data", "/data/../etc/x", "data/x", "/data/a,b"]:
        with pytest.raises(ValueError):
            runner.run("", [InputMount(host_path=src, container_path=bad)])
    with pytest.raises(FileNotFoundError):
        runner.run("", [InputMount(host_path=WATER / "missing.csv", container_path="/data/m")])
