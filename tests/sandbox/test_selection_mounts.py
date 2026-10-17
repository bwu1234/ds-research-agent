"""Growing a run's mounts (the end-to-end selection) on real containers.

A container cannot gain bind mounts, so ``DockerWorkspace.set_inputs`` ends
the session; the next cell runs in a new container with the new mounts.
Checked here: unselected siblings never exist under /data, state is lost on
growth, each session and rerun records its own mounts, and a selection the
size of the largest benchmark group (715 files) starts.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ds_research_agent.agent import DockerWorkspace
from ds_research_agent.sandbox import InputMount, SandboxRunner

pytestmark = pytest.mark.docker


def _files(root: Path, n: int) -> list[InputMount]:
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        (root / f"f{i}.csv").write_text(f"x\n{i}\n")
        out.append(InputMount(host_path=root / f"f{i}.csv", container_path=f"/data/d/f{i}.csv"))
    return out


async def test_growth_mounts_only_the_selection_and_restarts(
    runner: SandboxRunner, tmp_path: Path
) -> None:
    a, b, c = _files(tmp_path / "lake", 3)
    ws = DockerWorkspace(runner, [])
    await ws.set_inputs([a])
    first = await ws.execute(
        "import os\nx = 1\nprint(sorted(os.listdir('/data/d')), os.path.exists('/data/d/f1.csv'))",
        5,
    )
    assert first.status == "ok" and first.stdout.strip() == "['f0.csv'] False"
    await ws.set_inputs([a, b])
    second = await ws.execute(
        "import os\nprint('x' in globals(), sorted(os.listdir('/data/d')))\n"
        "print(open('/data/d/f1.csv').read().split()[1])",
        5,
    )
    assert second.status == "ok"
    assert second.stdout.split("\n")[:2] == ["False ['f0.csv', 'f1.csv']", "1"]
    assert [r.container_path for r in second.observed_reads] == ["/data/d/f1.csv"]
    run = await ws.rerun("print(open('/data/d/f1.csv').read())")
    assert run.inputs == ("/data/d/f0.csv", "/data/d/f1.csv")
    ends = await ws.close()
    assert [e.inputs for e in ends] == [("/data/d/f0.csv",), ("/data/d/f0.csv", "/data/d/f1.csv")]
    assert all(e.audit.complete for e in ends)
    assert c.container_path not in run.inputs


async def test_unchanged_selection_keeps_the_session(runner: SandboxRunner, tmp_path: Path) -> None:
    (a,) = _files(tmp_path / "lake", 1)
    ws = DockerWorkspace(runner, [a])
    await ws.execute("x = 1", 5)
    await ws.set_inputs([a])
    again = await ws.execute("print('x' in globals())", 5)
    assert again.stdout.strip() == "True"
    assert len(await ws.close()) == 1


async def test_largest_group_mounts(runner: SandboxRunner, tmp_path: Path) -> None:
    mounts = _files(tmp_path / "lake", 715)
    ws = DockerWorkspace(runner, mounts)
    t0 = time.monotonic()
    cell = await ws.execute("import os\nprint(len(os.listdir('/data/d')))", 30)
    started = time.monotonic() - t0
    assert cell.status == "ok" and cell.stdout.strip() == "715"
    (end,) = await ws.close()
    assert len(end.inputs) == 715 and end.audit.complete
    print(f"715 mounts: first cell after {started:.1f} s")
