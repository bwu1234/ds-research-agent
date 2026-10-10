"""Persistent kernel sessions against real containers (marker ``docker``)."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from ds_research_agent.sandbox import InputMount, KernelSession, SandboxRunner

pytestmark = pytest.mark.docker

WATER = Path(__file__).resolve().parents[1] / "fixtures/lake/environment/water"
A = "/data/water/readings.csv"
INPUTS = [InputMount(host_path=WATER / "station_readings.csv", container_path=A)]


@pytest.fixture
def k(runner: SandboxRunner) -> Iterator[KernelSession]:
    with KernelSession(runner, INPUTS) as session:
        yield session


def run(k: KernelSession, code: str, **kw: int) -> str:
    c = k.execute(textwrap.dedent(code), **kw)
    assert c.status == "ok", (c.status, c.traceback, c.stderr)
    return c.stdout


def test_state_persists_and_last_expression_echoes(k: KernelSession) -> None:
    run(k, f"import pandas as pd\ndf = pd.read_csv({A!r}, sep=';')")
    assert run(k, "len(df)") == "5\n"
    assert run(k, "x = 1") == ""
    assert run(k, "def f(v): return v * 2\nf(21)") == "42\n"


def test_cell_functions_pickle(k: KernelSession) -> None:
    out = run(
        k,
        """
        import pickle
        class P:
            pass
        def g():
            return 1
        len(pickle.dumps(g)) > 0 and len(pickle.dumps(P())) > 0
    """,
    )
    assert out == "True\n"


def test_errors_keep_the_kernel_and_trim_the_traceback(k: KernelSession) -> None:
    run(k, "y = 7")
    c = k.execute("def h():\n    return 1 / 0\nh()")
    assert c.status == "error" and c.kernel_alive
    assert c.traceback is not None
    assert "ZeroDivisionError" in c.traceback and '"<cell 2>", line 3' in c.traceback
    assert "kernel.py" not in c.traceback
    s = k.execute("def (:")
    assert s.status == "error" and "SyntaxError" in (s.traceback or "")
    assert k.execute("raise SystemExit(3)").status == "error"
    assert run(k, "y") == "7\n"


def test_output_is_captured_at_fd_level(k: KernelSession) -> None:
    c = k.execute("import os, sys\nprint('py')\nos.system('echo child; echo err >&2')\n"
                  "print('e', file=sys.stderr)")  # fmt: skip
    assert c.stdout == "py\nchild\n" and c.stderr == "err\ne\n"


def test_cells_cannot_read_requests_from_stdin(k: KernelSession) -> None:
    assert run(k, "import sys\nsys.stdin.read()") == "''\n"


def test_reads_attributed_per_cell_and_merged_per_session(k: KernelSession) -> None:
    c1 = k.execute(f"open({A!r}).read(); None")
    c2 = k.execute("1 + 1")
    c3 = k.execute(f"import subprocess\nsubprocess.run(['cat', {A!r}], capture_output=True); None")
    assert c1.audit.data_reads == [A] and c2.audit.data_reads == [] and c3.audit.data_reads == [A]
    assert c1.observed_reads[0].host_path == (WATER / "station_readings.csv").resolve()
    end = k.close()
    assert end.audit.complete and end.audit.data_reads == [A] and end.cells == 3


def test_detached_read_lands_in_session_audit(k: KernelSession) -> None:
    run(
        k,
        f"""
        import subprocess
        subprocess.Popen(["sh", "-c", "sleep 1; cat {A} >/dev/null"], start_new_session=True)
    """,
    )
    end = k.close()
    assert end.audit.complete and end.audit.data_reads == [A]


def test_timeout_interrupts_and_keeps_state(k: KernelSession) -> None:
    run(k, "z = 'kept'")
    c = k.execute("import time\nwhile True: time.sleep(0.1)", timeout_s=2)
    assert c.status == "timeout_interrupted" and c.kernel_alive
    assert "KeyboardInterrupt" in (c.traceback or "")
    assert run(k, "z") == "'kept'\n"


def test_uninterruptible_cell_kills_the_kernel_but_audit_completes(k: KernelSession) -> None:
    c = k.execute(
        "import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True: pass",
        timeout_s=2,
    )
    assert c.status == "timeout" and not c.kernel_alive
    assert k.execute("1").status == "dead"
    end = k.close()
    # The tracees were killed, not strace, so nothing ran unobserved.
    assert end.audit.complete, end.audit.issues


def test_kernel_crash_is_dead(k: KernelSession) -> None:
    c = k.execute("import os\nos._exit(1)")
    assert c.status == "dead" and not c.kernel_alive
    assert k.close().audit.complete


def test_out_of_memory_is_dead(k: KernelSession) -> None:
    c = k.execute("b = []\nwhile True: b.append(bytearray(64 * 1024 * 1024))", timeout_s=30)
    assert c.status in ("dead", "error"), c.status  # OOM kill, or MemoryError first


def test_forged_kernel_reply_is_a_protocol_error(k: KernelSession) -> None:
    c = k.execute(
        textwrap.dedent("""
        import gc, io
        pipes = [o for o in gc.get_objects() if isinstance(o, io.FileIO) and o.mode == "wb"]
        for p in pipes:
            p.write(b'{"cell": 99, "status": "ok", "traceback": null}\\n')
    """)
    )
    assert c.status == "protocol_error" and not c.kernel_alive


def test_session_isolation_matches_program_runs(k: KernelSession) -> None:
    out = run(
        k,
        """
        import json, os, socket
        res = {"uid": os.getuid(), "data": sorted(os.listdir("/data/water"))}
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=2); res["net"] = True
        except OSError:
            res["net"] = False
        try:
            open("/data/water/x", "w"); res["ro"] = False
        except OSError:
            res["ro"] = True
        json.dumps(res)
    """,
    )
    assert out.strip() == repr('{"uid": 1000, "data": ["readings.csv"], "net": false, "ro": true}')


def test_session_timeout(runner: SandboxRunner) -> None:
    short = SandboxRunner(runner.settings.model_copy(update={"session_timeout_s": 4}))
    with KernelSession(short, INPUTS) as k:
        c = k.execute("import time\ntime.sleep(30)", timeout_s=30)
        assert c.status == "timeout_interrupted"  # cut at the session's 4 s
        assert k.execute("1").status == "dead"
        end = k.close()
    assert end.session_timed_out and end.audit.complete
    assert end.wall_s < 30
