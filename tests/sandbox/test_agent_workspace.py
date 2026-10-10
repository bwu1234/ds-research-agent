"""The agent loop on a real kernel and rerun, driven by a scripted model.

No LLM: the model is a script, so these check the loop, workspace, and
verification wiring against real containers. They also demonstrate the
verification limits the D3 acceptance asks for: a wrong calculation and a
hard-coded answer both reproduce and pass observed access (only the
evaluator can catch them), while a program that never reads the data fails
access.
"""

from __future__ import annotations

from typing import Any

import pytest

from ds_research_agent.agent import SYSTEM_PROMPT, DockerWorkspace, Outcome, run_agent, user_prompt
from ds_research_agent.sandbox import InputMount, SandboxRunner
from tests.sandbox.test_kernel import INPUTS, A
from tests.test_agent_loop import SETTINGS, Rec, Script, py, reply
from tests.test_agent_loop import submit as _submit

pytestmark = pytest.mark.docker

READ = f"import pandas as pd\ndf = pd.read_csv({A!r}, sep=';')\n"


async def go(runner: SandboxRunner, *replies: Any, inputs: list[InputMount] = INPUTS) -> Outcome:
    client = Script(*replies)
    out = await run_agent(
        client,
        DockerWorkspace(runner, inputs),
        Rec(),
        system=SYSTEM_PROMPT,
        user=user_prompt("How many readings?", [(A, 100)], None),
        files=[A],
        settings=SETTINGS,
        cell_timeout_s=5,
    )
    assert all(s.audit.complete for s in out.sessions)
    return out


def submit(answer: Any, program: str, files: tuple[str, ...] = (A,)) -> tuple[str, dict[str, Any]]:
    return _submit(answer, files=files, program=program)  # type: ignore[no-any-return]


def program(expr: str, read: bool = True) -> str:
    head = READ if read else ""
    return head + f'import json\nprint(json.dumps({{"answer": {expr}}}))\n'


async def test_explore_then_submit_verifies(runner: SandboxRunner) -> None:
    out = await go(
        runner, reply(py(READ + "len(df)")), reply(submit(5, program=program("len(df)")))
    )
    v = out.verification
    assert out.stop_reason == "submitted" and v is not None
    assert v.reproduced and v.access_verified and v.observed_files == (A,)


async def test_kernel_state_persists_across_steps(runner: SandboxRunner) -> None:
    out = await go(
        runner,
        reply(py(READ)),
        reply(py("n = int(df['ph'].notna().sum())\nn")),
        reply(submit(5, program=program("int(df['ph'].notna().sum())"))),
    )
    assert out.verification is not None and out.verification.reproduced


async def test_wrong_calculation_still_reproduces(runner: SandboxRunner) -> None:
    # Off by one on purpose: reproduction and access pass; correctness is
    # the evaluator's call.
    out = await go(runner, reply(submit(4, program=program("len(df) - 1"))))
    v = out.verification
    assert v is not None and v.reproduced and v.access_verified


async def test_hard_coded_answer_that_reads_the_file_passes(runner: SandboxRunner) -> None:
    out = await go(runner, reply(submit(42, program=program("42"))))
    v = out.verification
    assert v is not None and v.reproduced and v.access_verified


async def test_program_that_reads_nothing_fails_access(runner: SandboxRunner) -> None:
    out = await go(runner, reply(submit(42, files=(), program=program("42", read=False))))
    v = out.verification
    assert v is not None and v.reproduced and not v.access_verified
    assert "rerun read no data files" in v.detail


async def test_program_relying_on_kernel_state_fails_reproduction(runner: SandboxRunner) -> None:
    out = await go(
        runner,
        reply(py(READ + "open('/scratch/n.txt', 'w').write(str(len(df)))")),
        reply(
            submit(
                5,
                program=READ
                + "import json\nprint(json.dumps({'answer': int(open('/scratch/n.txt').read())}))",
            )
        ),  # noqa: E501
    )
    v = out.verification
    assert v is not None and not v.reproduced and v.rerun_exit_code != 0


async def test_dead_kernel_restarts_with_fresh_state(runner: SandboxRunner) -> None:
    out = await go(
        runner,
        reply(py("x = 1\nimport os\nos._exit(1)")),
        reply(py("print('x' in globals())")),
        reply(submit(5, program=program("len(df)"))),
    )
    assert out.stop_reason == "submitted" and len(out.sessions) == 2
