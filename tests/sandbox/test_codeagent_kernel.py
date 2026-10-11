"""The comparator's executor on a real kernel: no model, no smolagents loop.

Checks the one part the offline fake cannot: that ``final_answer`` defined
by the setup cell comes back through the real kernel's traceback intact
(numpy values, multi-line programs), that ordinary errors come back as
errors, and that a payload cut off by the bridge's traceback limit is an
error rather than a wrong submission.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from ds_research_agent.agent import DockerWorkspace, ProgramEvent, verification
from ds_research_agent.agent.answers import Submission
from ds_research_agent.sandbox import SandboxRunner
from eval.kramabench.codeagent_runs import SandboxExecutor, _Bridge, _CellFailed
from tests.sandbox.test_kernel import INPUTS, A
from tests.test_agent_loop import SETTINGS

pytestmark = pytest.mark.docker

READ = f"import pandas as pd\ndf = pd.read_csv({A!r}, sep=';')\n"
PROGRAM = READ + 'import json\nprint(json.dumps({"answer": len(df)}))\n'


class Rec:
    def __init__(self) -> None:
        self.events: list[ProgramEvent] = []

    def program(self, e: ProgramEvent) -> None:
        self.events.append(e)


async def test_final_answer_round_trips_through_the_kernel(runner: SandboxRunner) -> None:
    ws = DockerWorkspace(runner, INPUTS)
    rec = Rec()
    bridge = _Bridge(asyncio.get_running_loop(), 120, time.monotonic)
    settings = SETTINGS.model_copy(update={"max_tool_output_chars": 4000})
    ex = SandboxExecutor(ws, rec, bridge, lambda: 1, settings, 5)  # type: ignore[arg-type]

    def call(code: str) -> Any:
        return asyncio.to_thread(ex, code)

    try:
        out = await call(READ + "print(len(df))")
        assert not out.is_final_answer and out.logs.strip() == "5"
        with pytest.raises(_CellFailed, match="ZeroDivisionError"):
            await call("1 / 0")
        out = await call(
            "import numpy as np\n"
            f"final_answer(answer=np.int64(len(df)), files_used=[{A!r}], program={PROGRAM!r},"
            " assumptions=['semicolon separated'])"
        )
        assert out.is_final_answer
        assert out.output == {
            "answer": 5,
            "files_used": [A],
            "program": PROGRAM,
            "assumptions": ["semicolon separated"],
        }
        # Past the bridge's traceback limit (2,000 chars in these settings):
        # an error for the model, not a truncated submission.
        with pytest.raises(_CellFailed):
            await call(f"final_answer(answer=1, files_used=[{A!r}], program='x' * 5000)")
        assert [e.kind for e in rec.events] == ["setup", "cell", "cell", "cell", "cell"]
        assert len({e.session for e in rec.events}) == 1  # setup once per session

        sub = Submission(answer=5, files_used=(A,), program=PROGRAM, assumptions=())
        v = verification(sub, await ws.rerun(PROGRAM), None)
        assert v.reproduced and v.access_verified
    finally:
        ends = await ws.close()
    assert all(e.audit.complete for e in ends)
