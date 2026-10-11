"""The D3 minimal comparator: smolagents ``CodeAgent`` on the given-files task.

CodeAgent keeps its own system prompt, code-block format, error retry text,
and step loop. What it shares with ``given_files`` is the environment: the
same labelled files mounted read-only, the same audited kernel and package
image, the same model and model settings, the same step, wall-clock, cell
and output budgets, the same submission contract, and the same verifier.
None of this condition's guidance (analysis tips, re-plan and no-progress
notes, budget notices, the submit check) is given to it.

Three seams make this work without changing smolagents:

- ``SandboxExecutor`` is CodeAgent's ``executor``: each code block is one
  cell in the run's ``Workspace`` (Docker kernel live, ledger in replay).
  A ``setup`` cell defines ``final_answer`` in the kernel; calling it raises
  an exception whose message is the submission as one JSON line, which the
  executor reads back from the cell's traceback.
- ``RecordedModel`` is CodeAgent's ``model``: requests go through the
  harness's ``ModelClient`` (the ``RecordingClient``), so every request is a
  ledger step and ``ReplayClient`` replays the run. CodeAgent's stop
  sequences are fixed model options (``STOP``) for this condition.
- ``final_answer_checks`` applies the submission schema and
  ``check_submission``; a rejected submission comes back as an error and the
  run continues, as a rejected ``submit_answer`` call does.

CodeAgent's code is synchronous, so it runs in a worker thread and calls
back into the harness's event loop for the model and the kernel.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, TypeVar

from jsonschema import Draft202012Validator
from smolagents import CodeAgent, Tool
from smolagents.local_python_executor import CodeOutput, PythonExecutor
from smolagents.models import ChatMessage as SmolMessage
from smolagents.models import (
    MessageRole,
    Model,
    get_clean_message_list,
    tool_role_conversions,
)
from smolagents.monitoring import LogLevel, TokenUsage
from smolagents.utils import AgentError, AgentGenerationError, truncate_content

from ds_research_agent.agent import (
    ProgramEvent,
    Submission,
    Workspace,
    check_submission,
    user_prompt,
    verification,
)
from ds_research_agent.agent.tools import SANDBOX_PACKAGES, SUBMIT_ANSWER
from ds_research_agent.config import AgentSettings
from ds_research_agent.ledger import Answer, Batch, Run, utc_now
from ds_research_agent.models import ChatMessage, ModelClient
from ds_research_agent.sandbox import (
    CellResult,
    SandboxError,
    SandboxRun,
    SandboxUnavailable,
)
from eval.kramabench.agent_runs import (
    ACCESS_POLICY,
    LedgerRecorder,
    RecordingClient,
    WorkspaceFor,
    task_inputs,
)
from eval.kramabench.baselines import Harness, score_run, sha256_text
from eval.kramabench.tasks import Task

CONDITION = "codeagent"
# CodeAgent's stop sequences for its default ``<code>`` tags, set as model
# options for this condition (the client takes no per-request options).
STOP = ["Observation:", "Calling tools:", "</code>"]
MARKER = "DSRA_FINAL_ANSWER "

# The environment, stated as plainly as in the given-files prompt; CodeAgent's
# own prompt supplies everything about how to work.
INSTRUCTIONS = (
    "Environment: your code runs in a Python 3.14 sandbox. The data files are "
    "read-only under /data, and /scratch is writable. There is no internet access "
    f"and no package installation. Available packages: {', '.join(SANDBOX_PACKAGES)}, "
    "and the standard library. Your final answer must be a JSON number, string, or "
    "list, with a self-contained program that recomputes it from the raw files; "
    "the program is rerun in a fresh sandbox to check that it reproduces the answer."
)

# Defines final_answer in the kernel. Calling it ends the cell with an
# exception; its message is the submission, one JSON line, at the end of the
# traceback (the bridge keeps the traceback's tail).
SETUP = f"""\
class _FinalAnswer(BaseException):
    pass


def final_answer(answer, files_used, program, assumptions=None):
    import json as _json

    def _plain(v):
        return v.item() if hasattr(v, "item") else v.tolist() if hasattr(v, "tolist") else str(v)

    raise _FinalAnswer({MARKER!r} + _json.dumps(
        {{"answer": answer, "files_used": files_used, "program": program,
         "assumptions": assumptions}}, default=_plain))
"""

_PROPS = SUBMIT_ANSWER.parameters["properties"]
_TYPES = {"answer": "any", "files_used": "array", "program": "string", "assumptions": "array"}


class FinalAnswerTool(Tool):  # type: ignore[misc]
    """``final_answer`` as CodeAgent's prompt shows it; it runs in the kernel
    (``SETUP``), never on the host."""

    name = "final_answer"
    description = (
        "Submit the final answer and end the task. The program is run in a fresh "
        "sandbox and must reproduce the answer."
    )
    inputs = {
        k: {"type": t, "description": _PROPS[k]["description"]}
        | ({"nullable": True} if k == "assumptions" else {})
        for k, t in _TYPES.items()
    }
    output_type = "any"

    def forward(  # noqa: D102
        self, answer: Any, files_used: Any, program: str, assumptions: Any = None
    ) -> Any:
        raise RuntimeError("final_answer runs in the sandbox kernel")


class _Stopped(Exception):
    """Ends the CodeAgent run from inside it with one of our stop reasons."""

    def __init__(self, reason: str, error: str | None = None) -> None:
        super().__init__(reason)
        self.reason, self.error = reason, error


class _StepsExhausted(Exception):
    pass


class _Agent(CodeAgent):  # type: ignore[misc]
    """CodeAgent, except that running out of steps ends with no answer, as in
    ``given_files``, instead of an extra code-free request for one."""

    def _handle_max_steps_reached(self, task: str) -> Any:
        raise _StepsExhausted()


T = TypeVar("T")


class _CellFailed(Exception):
    """A cell that failed: shown to the model, as CodeAgent shows any code error."""


class _Bridge:
    """Runs harness coroutines on the event loop from CodeAgent's thread,
    against one run's wall-clock budget. Ledger writes happen there too: the
    ledger's SQLite connection belongs to the loop's thread."""

    def __init__(
        self, loop: asyncio.AbstractEventLoop, max_wall_s: float, clock: Callable[[], float]
    ) -> None:
        self.loop, self.clock = loop, clock
        self.deadline = clock() + max_wall_s
        # CodeAgent turns exceptions from the executor into feedback for the
        # model and carries on, so a stop is recorded here, the agent is
        # interrupted (it raises at the next step), and the run ends with it.
        self.stop: _Stopped | None = None
        self.on_halt: Callable[[], None] = lambda: None
        # Not the run's fault (Docker down, a harness bug): re-raised after the run.
        self.fatal: BaseException | None = None

    def left(self) -> float:
        return self.deadline - self.clock()

    def call(self, coro: Coroutine[Any, Any, T]) -> T:
        left = self.left()
        if self.stop is not None or left <= 0:
            coro.close()
            raise self.stop or self.halt(_Stopped("max_wall"))
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout=left)
        except TimeoutError:
            fut.cancel()
            raise self.halt(_Stopped("max_wall")) from None

    def halt(self, e: _Stopped) -> _Stopped:
        if self.stop is None:
            self.stop = e
            self.on_halt()
        return self.stop

    def harness_failed(self, e: BaseException) -> _Stopped:
        self.fatal = self.fatal or e
        return self.halt(_Stopped("harness_error", f"{type(e).__name__}: {e}"))


class SandboxExecutor(PythonExecutor):  # type: ignore[misc]
    """CodeAgent's executor on the run's kernel workspace."""

    def __init__(
        self,
        workspace: Workspace,
        recorder: LedgerRecorder,
        bridge: _Bridge,
        step: Callable[[], int],
        settings: AgentSettings,
        cell_timeout_s: int,
    ) -> None:
        self.ws, self.recorder, self.bridge, self.step = workspace, recorder, bridge, step
        self.s, self.cell_timeout_s = settings, cell_timeout_s
        self.seq = 0
        self.ready_in: str | None = None  # the kernel session that ran SETUP
        self.state: dict[str, Any] = {}

    def send_tools(self, tools: dict[str, Tool]) -> None:
        if set(tools) != {"final_answer"}:
            raise ValueError(f"only final_answer is supported, got {sorted(tools)}")

    def send_variables(self, variables: dict[str, Any]) -> None:
        if variables:
            raise ValueError("variables are not sent to the kernel")

    async def _execute(self, code: str, kind: str, step: int, timeout: int) -> CellResult:
        cell = await self.ws.execute(code, timeout)
        self.recorder.program(
            ProgramEvent(
                seq=self.seq,
                step=step,
                call_index=0 if kind == "cell" else -1,
                kind=kind,  # type: ignore[arg-type]
                session=self.ws.session_id,
                code=code,
                cell=cell,
            )
        )
        self.seq += 1
        return cell

    def _cell(self, code: str, kind: str) -> CellResult:
        timeout = max(1, min(self.cell_timeout_s, int(self.bridge.left())))
        try:
            return self.bridge.call(self._execute(code, kind, self.step(), timeout))
        except SandboxUnavailable as e:
            self.bridge.fatal = e
            raise self.bridge.halt(_Stopped("sandbox_error", str(e))) from e
        except SandboxError as e:
            raise self.bridge.halt(_Stopped("sandbox_error", str(e))) from e

    def _setup(self) -> None:
        """Define final_answer in the current kernel session (again after a restart)."""
        if self.ready_in is not None and self.ready_in == self.ws.session_id:
            return
        cell = self._cell(SETUP, "setup")
        if cell.status != "ok":
            raise self.bridge.halt(_Stopped("sandbox_error", f"setup cell {cell.status}"))
        self.ready_in = self.ws.session_id

    def __call__(self, code_action: str) -> CodeOutput:
        try:
            return self._run(code_action)
        except _CellFailed, _Stopped:
            raise
        except Exception as e:
            raise self.bridge.harness_failed(e) from e

    def _run(self, code: str) -> CodeOutput:
        self._setup()
        cell = self._cell(code, "cell")
        if not cell.kernel_alive:
            self.ready_in = None
        limit = self.s.max_tool_output_chars
        logs = truncate_content(cell.stdout + cell.stderr, max_length=limit)
        if cell.status == "ok":
            return CodeOutput(output=None, logs=logs, is_final_answer=False)
        last = (cell.traceback or "").rstrip("\n").rsplit("\n", 1)[-1]
        at = last.find(MARKER)
        if cell.status == "error" and at >= 0:
            try:
                return CodeOutput(
                    output=json.loads(last[at + len(MARKER) :]), logs=logs, is_final_answer=True
                )
            except json.JSONDecodeError:
                pass  # cut off: reported below as the error it is
        # CodeAgent shows _print_outputs as the logs of a failed cell.
        self.state["_print_outputs"] = logs
        tb = truncate_content(cell.traceback or "", max_length=limit)
        if not cell.kernel_alive:
            tb += f"\nThe Python kernel was restarted ({cell.status}); its variables are lost."
        raise _CellFailed(f"Code execution failed ({cell.status}):\n{tb}")


class RecordedModel(Model):  # type: ignore[misc]
    """CodeAgent's model on the harness's client (recorded, or replayed)."""

    def __init__(self, client: ModelClient, bridge: _Bridge, stop: Sequence[str]) -> None:
        super().__init__(model_id="harness")
        self.client, self.bridge, self.stop = client, bridge, list(stop)

    def generate(
        self,
        messages: list[SmolMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> SmolMessage:
        if list(stop_sequences or ()) != self.stop or response_format or tools_to_call_from:
            raise self.bridge.harness_failed(
                ValueError("request outside this condition's fixed settings")
            )
        try:
            flat = get_clean_message_list(
                messages, role_conversions=tool_role_conversions, flatten_messages_as_text=True
            )
            ours = [ChatMessage(role=m["role"], content=m["content"]) for m in flat]
        except Exception as e:
            raise self.bridge.harness_failed(e) from e
        try:
            result = self.bridge.call(self.client.chat(ours))
        except _Stopped:
            raise
        except Exception as e:  # noqa: BLE001 - becomes the run's stop reason
            raise self.bridge.halt(_Stopped("model_error", f"{type(e).__name__}: {e}")) from e
        u = result.usage
        return SmolMessage(
            role=MessageRole.ASSISTANT,
            content=result.message.content,
            token_usage=TokenUsage(
                input_tokens=u.prompt_eval_tokens or 0, output_tokens=u.output_tokens or 0
            ),
        )


def _submission_check(files: Sequence[str]) -> Callable[..., bool]:
    validator = Draft202012Validator(SUBMIT_ANSWER.parameters)

    def submission_valid(answer: Any, memory: Any, agent: Any = None) -> bool:
        if not isinstance(answer, dict):
            raise ValueError("final_answer was not called with the submission arguments")
        args = {k: v for k, v in answer.items() if not (k == "assumptions" and v is None)}
        problems = [e.message for e in validator.iter_errors(args)]
        problems += check_submission(args, files)
        if problems:
            raise ValueError("; ".join(problems) + ". Nothing was submitted.")
        return True

    return submission_valid


def build_agent(
    model: Model, executor: PythonExecutor, files: Sequence[str], max_steps: int
) -> CodeAgent:
    return _Agent(
        tools=[FinalAnswerTool()],
        model=model,
        executor=executor,
        additional_authorized_imports=["*"],
        instructions=INSTRUCTIONS,
        max_steps=max_steps,
        final_answer_checks=[_submission_check(files)],
        verbosity_level=LogLevel.OFF,
    )


class _NoModel(Model):  # type: ignore[misc]
    def generate(self, *a: Any, **k: Any) -> SmolMessage:
        raise RuntimeError("no model")


def prefix_sha256() -> str:
    """Identity of the cached prefix: CodeAgent's rendered system prompt and the
    stop options. The file list is in the task message, not the prompt."""

    class _Idle(PythonExecutor):  # type: ignore[misc]
        def send_tools(self, tools: dict[str, Tool]) -> None: ...
        def send_variables(self, variables: dict[str, Any]) -> None: ...
        def __call__(self, code_action: str) -> CodeOutput:
            raise RuntimeError

    agent = build_agent(_NoModel(), _Idle(), (), 1)
    return sha256_text(agent.system_prompt + json.dumps(STOP))


def run_codeagent(agent: CodeAgent, task: str, bridge: _Bridge) -> tuple[str, Any, str | None]:
    """(stop reason, final answer dict or None, error) for one CodeAgent run."""
    try:
        out = agent.run(task)
    except _StepsExhausted:
        return "max_steps", None, None
    except AgentError as e:
        stop = bridge.stop
        if stop is not None:
            return stop.reason, None, stop.error
        if isinstance(e, AgentGenerationError):
            return "model_error", None, str(e)
        raise
    return "submitted", out, None


async def run_codeagent_task(
    h: Harness,
    batch: Batch,
    task: Task,
    repeat: int,
    client: ModelClient,
    workspace_for: WorkspaceFor,
    image_id: str | None,
    clock: Callable[[], float] = time.monotonic,
) -> Run:
    mounts, listed, manifest = task_inputs(h, task)
    manifest["sandbox_image"] = image_id
    run = Run(
        run_id=f"{batch.batch_id}/{task.key}/{repeat}",
        batch_id=batch.batch_id,
        task_key=task.key,
        parent_task_key=task.parent_key,
        split=h.split.split_of(task.key),
        condition=CONDITION,
        repeat=repeat,
        input_manifest=manifest,
        access_policy=ACCESS_POLICY,
        catalogue_generation=None,
        started=utc_now(),
    )
    h.ledger.add_run(run)
    rc = RecordingClient(client, h, run.run_id)
    recorder = LedgerRecorder(h.ledger, run.run_id)
    workspace = workspace_for(task, repeat, mounts)
    s = h.settings.agent
    files = [c for c, _ in listed]
    visible_at = task.answer_type if h.settings.eval.answer_type_visible else None
    start = clock()
    bridge = _Bridge(asyncio.get_running_loop(), s.max_wall_s, clock)
    executor = SandboxExecutor(
        workspace, recorder, bridge, lambda: agent.step_number, s, h.settings.sandbox.cell_timeout_s
    )
    agent = build_agent(RecordedModel(rc, bridge, STOP), executor, files, s.max_steps)
    bridge.on_halt = agent.interrupt
    prompt = user_prompt(task.query, listed, visible_at)
    sub: Submission | None = None
    try:
        reason, out, error = await asyncio.to_thread(run_codeagent, agent, prompt, bridge)
        if rc.fatal is not None:
            raise rc.fatal
        if bridge.fatal is not None:
            raise bridge.fatal
        steps = agent.step_number - 1
        if out is not None:
            sub = Submission(
                answer=out["answer"],
                files_used=tuple(out["files_used"]),
                program=out["program"],
                assumptions=tuple(out.get("assumptions") or ()),
            )
            recorder.submission(sub, steps)
            try:
                rerun: SandboxRun | None = await workspace.rerun(sub.program)
                rerun_error = None
            except SandboxUnavailable:
                raise
            except SandboxError as e:
                rerun, rerun_error = None, f"rerun failed in the sandbox: {e}"
            if rerun is not None:
                recorder.program(
                    ProgramEvent(
                        seq=executor.seq,
                        step=steps,
                        call_index=0,
                        kind="final_rerun",
                        session=rerun.container,
                        code=sub.program,
                        run=rerun,
                    )
                )
            recorder.verification(verification(sub, rerun, rerun_error))
    finally:
        sessions = await workspace.close()
    for end in sessions:
        recorder.session_end(end)
    h.ledger.put_answer(
        Answer(
            run_id=run.run_id,
            answered=sub is not None,
            value=sub.answer if sub else None,
            parse_status="ok" if sub else reason,
        )
    )
    score_run(h.ledger, run.run_id, task, CONDITION)
    h.ledger.finish_run(
        run.run_id,
        ended=utc_now(),
        wall_s=round(clock() - start, 3),
        stop_reason=reason,
        error=error,
    )
    return run
