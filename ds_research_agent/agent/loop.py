"""The given-files analysis loop: model step, run code, repeat, submit, rerun.

Each step is one model turn through the bounded tool-call repair policy;
the conversation is append-only, so Ollama can reuse the cached prefix.
``run_python`` cells go to the workspace's persistent kernel and come back
as bounded text. ``submit_answer`` is checked, then its program is rerun in
a fresh sandbox and compared with ``reproduction-v1``. Budgets (steps, wall
clock) end the run with a stop reason and no answer; the loop never guesses.

The loop never sees gold answers: scoring is the evaluator's job. Model
requests are recorded by the client the caller passes in (the harness wraps
its client); programs, the submission, and the verification go to the
``Recorder``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ds_research_agent.agent.answers import (
    COMPARATOR,
    Submission,
    check_submission,
    program_answer,
    reproduces,
)
from ds_research_agent.agent.tool_calls import chat_with_repair
from ds_research_agent.agent.tools import (
    BUDGET_LOW,
    CUT_OFF,
    KERNEL_RESTARTED,
    LAST_TURN,
    NO_TOOL_CALL,
    REPLAN,
    TOOLS,
)
from ds_research_agent.agent.workspace import Workspace
from ds_research_agent.config import AgentSettings
from ds_research_agent.models import ChatMessage, ModelClient, ToolCall
from ds_research_agent.sandbox import (
    CellResult,
    SandboxError,
    SandboxRun,
    SandboxUnavailable,
    SessionEnd,
)

StopReason = Literal[
    "submitted", "max_steps", "max_wall", "tool_call_failure", "model_error", "sandbox_error"
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProgramEvent(_Frozen):
    """One exploration cell or the final program's fresh rerun."""

    seq: int
    step: int  # the model step that called it; for the rerun, the submit step
    call_index: int
    kind: Literal["cell", "final_rerun"]
    session: str | None
    code: str
    cell: CellResult | None = None
    run: SandboxRun | None = None
    replan_requested: bool = False


class Verification(_Frozen):
    comparator: str
    rerun_exit_code: int | None
    rerun_answered: bool
    rerun_answer: Any
    reproduced: bool
    audit_complete: bool
    observed_files: tuple[str, ...]
    claimed_files: tuple[str, ...]
    # Complete audit, at least one data read, and observed == claimed.
    access_verified: bool
    detail: tuple[str, ...]


class Outcome(_Frozen):
    stop_reason: StopReason
    steps: int
    wall_s: float
    submission: Submission | None
    verification: Verification | None
    error: str | None = None
    sessions: tuple[SessionEnd, ...] = ()


class Recorder(Protocol):
    def program(self, event: ProgramEvent) -> None: ...

    def submission(self, submission: Submission, step: int) -> None: ...

    def verification(self, verification: Verification) -> None: ...


def _head(text: str, limit: int, truncated: bool) -> str:
    if len(text) > limit:
        text, truncated = text[:limit], True
    return text + ("\n[truncated]" if truncated else "")


def _tail(text: str, limit: int, truncated: bool) -> str:
    if len(text) > limit:
        text, truncated = text[-limit:], True
    return ("[truncated]\n" if truncated else "") + text


def render_cell(cell: CellResult, limit: int) -> str:
    """A cell result as the model sees it; empty sections are left out."""
    parts = [f"[status: {cell.status}]"]
    if cell.stdout or cell.stdout_truncated:
        parts.append("stdout:\n" + _head(cell.stdout, limit, cell.stdout_truncated))
    if cell.stderr or cell.stderr_truncated:
        parts.append("stderr:\n" + _tail(cell.stderr, limit, cell.stderr_truncated))
    if cell.traceback:
        parts.append("traceback:\n" + _tail(cell.traceback.rstrip("\n"), limit, False))
    if len(parts) == 1 and cell.status == "ok":
        parts[0] = "[status: ok, no output]"
    return "\n".join(parts)


def error_signature(cell: CellResult) -> str | None:
    """What makes two failures the same error; None for a successful cell."""
    if cell.status == "ok":
        return None
    lines = [line for line in (cell.traceback or "").splitlines() if line.strip()]
    return f"{cell.status}: {lines[-1].strip()}" if lines else cell.status


def _tool(name: str, content: str) -> ChatMessage:
    return ChatMessage(role="tool", tool_name=name, content=content)


class _Run:
    def __init__(
        self,
        client: ModelClient,
        workspace: Workspace,
        recorder: Recorder,
        files: Sequence[str],
        settings: AgentSettings,
        cell_timeout_s: int,
        clock: Callable[[], float],
    ) -> None:
        self.client, self.workspace, self.recorder = client, workspace, recorder
        self.files = list(files)
        self.s = settings
        self.cell_timeout_s = cell_timeout_s
        self.clock = clock
        self.start = clock()
        self.seq = 0
        self.last_error: str | None = None
        self.repeats = 0

    def left(self) -> float:
        return self.s.max_wall_s - (self.clock() - self.start)

    async def run_cell(self, step: int, index: int, call: ToolCall) -> ChatMessage:
        code = str(call.arguments["code"])
        timeout = max(1, min(self.cell_timeout_s, int(self.left())))
        cell = await self.workspace.execute(code, timeout)
        sig = error_signature(cell)
        if sig is not None and sig == self.last_error:
            self.repeats += 1
        else:
            self.repeats = 1 if sig is not None else 0
        self.last_error = sig
        replan = sig is not None and self.repeats >= self.s.replan_after_repeats
        content = render_cell(cell, self.s.max_tool_output_chars)
        if not cell.kernel_alive:
            content += "\n" + KERNEL_RESTARTED.format(why=cell.status)
        if replan:
            content += "\n" + REPLAN.format(n=self.repeats)
        self.recorder.program(
            ProgramEvent(
                seq=self.seq,
                step=step,
                call_index=index,
                kind="cell",
                session=self.workspace.session_id,
                code=code,
                cell=cell,
                replan_requested=replan,
            )
        )
        self.seq += 1
        return _tool("run_python", content)

    async def verify(self, sub: Submission, step: int, index: int) -> Verification:
        detail: list[str] = []
        run: SandboxRun | None = None
        try:
            run = await self.workspace.rerun(sub.program)
        except SandboxUnavailable:
            raise
        except SandboxError as e:
            detail.append(f"rerun failed in the sandbox: {e}")
        if run is not None:
            self.recorder.program(
                ProgramEvent(
                    seq=self.seq,
                    step=step,
                    call_index=index,
                    kind="final_rerun",
                    session=run.container,
                    code=sub.program,
                    run=run,
                )
            )
            self.seq += 1
        answered, value = (False, None)
        if run is not None and run.exit_code == 0:
            answered, value = program_answer(run.stdout)
        if run is not None and run.exit_code != 0:
            detail.append(f"rerun exited {run.exit_code}")
        if run is not None and run.exit_code == 0 and not answered:
            detail.append('rerun printed no {"answer": ...} last line')
        reproduced = answered and reproduces(sub.answer, value)
        if answered and not reproduced:
            detail.append("rerun answer differs from the submitted answer")
        observed = tuple(sorted(o.container_path for o in run.observed_reads)) if run else ()
        claimed = tuple(sorted(set(sub.files_used)))
        complete = bool(run and run.audit.complete)
        if run is not None and not complete:
            detail.append("read audit incomplete: " + "; ".join(run.audit.issues))
        if complete and not observed:
            detail.append("rerun read no data files")
        if complete and observed and set(observed) != set(claimed):
            detail.append("observed reads differ from files_used")
        return Verification(
            comparator=COMPARATOR,
            rerun_exit_code=run.exit_code if run else None,
            rerun_answered=answered,
            rerun_answer=value,
            reproduced=reproduced,
            audit_complete=complete,
            observed_files=observed,
            claimed_files=claimed,
            access_verified=complete and bool(observed) and set(observed) == set(claimed),
            detail=tuple(detail),
        )

    async def run(self, messages: list[ChatMessage]) -> Outcome:
        steps = 0
        while True:
            if steps >= self.s.max_steps:
                return self.outcome("max_steps", steps)
            left = self.left()
            if left <= 0:
                return self.outcome("max_wall", steps)
            try:
                step = await asyncio.wait_for(
                    chat_with_repair(
                        self.client,
                        messages,
                        TOOLS,
                        self.s.tool_call_max_repairs,
                        self.s.server_retries,
                    ),
                    timeout=left,
                )
            except TimeoutError:
                return self.outcome("max_wall", steps)
            except Exception as e:  # noqa: BLE001 - recorded as the stop reason
                return self.outcome("model_error", steps, f"{type(e).__name__}: {e}")
            steps += 1
            messages += step.appended
            if step.result is None:
                return self.outcome("tool_call_failure", steps)
            result = step.result
            calls = result.message.tool_calls
            if not calls:
                text = CUT_OFF if result.done_reason == "length" else NO_TOOL_CALL
                messages.append(ChatMessage(role="user", content=text))
                self.budget_notice(messages, steps)
                continue
            replies: list[ChatMessage] = []
            submitted: tuple[Submission, int] | None = None
            try:
                for i, call in enumerate(calls):
                    if submitted is not None:
                        replies.append(
                            _tool(call.name, "Not run: the answer was already submitted.")
                        )
                    elif call.name == "run_python":
                        replies.append(await self.run_cell(steps, i, call))
                    else:  # submit_answer; the repair policy rejects unknown tools
                        problems = check_submission(call.arguments, self.files)
                        if problems:
                            replies.append(
                                _tool(
                                    "submit_answer",
                                    "Error: " + "; ".join(problems) + ". Nothing was "
                                    "submitted. Fix this and call submit_answer again.",
                                )
                            )
                            continue
                        args = call.arguments
                        sub = Submission(
                            answer=args["answer"],
                            files_used=tuple(args["files_used"]),
                            program=args["program"],
                            assumptions=tuple(args.get("assumptions", ())),
                        )
                        submitted = (sub, i)
            except SandboxUnavailable:
                raise  # infrastructure, not the run: the batch stops, resumable
            except SandboxError as e:
                return self.outcome("sandbox_error", steps, str(e))
            messages += replies
            if submitted is None:
                self.budget_notice(messages, steps)
            else:
                sub, index = submitted
                self.recorder.submission(sub, steps)
                verification = await self.verify(sub, steps, index)
                self.recorder.verification(verification)
                return self.outcome("submitted", steps, submission=sub, verification=verification)

    def budget_notice(self, messages: list[ChatMessage], steps: int) -> None:
        """Append the fixed budget notice to the step's last message, which the
        model has not seen yet, so the context stays append-only."""
        left = self.s.max_steps - steps
        if not 0 < left <= self.s.budget_warning_steps:
            return
        notice = LAST_TURN if left == 1 else BUDGET_LOW.format(n=left)
        last = messages[-1]
        messages[-1] = last.model_copy(update={"content": last.content + "\n" + notice})

    def outcome(
        self,
        reason: StopReason,
        steps: int,
        error: str | None = None,
        *,
        submission: Submission | None = None,
        verification: Verification | None = None,
    ) -> Outcome:
        return Outcome(
            stop_reason=reason,
            steps=steps,
            wall_s=round(self.clock() - self.start, 3),
            submission=submission,
            verification=verification,
            error=error,
        )


async def run_agent(
    client: ModelClient,
    workspace: Workspace,
    recorder: Recorder,
    *,
    system: str,
    user: str,
    files: Sequence[str],
    settings: AgentSettings,
    cell_timeout_s: int,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Run one task to a stop reason. ``files`` are the container paths given."""
    r = _Run(client, workspace, recorder, files, settings, cell_timeout_s, clock)
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    try:
        out = await r.run(messages)
    finally:
        sessions = await workspace.close()
    return out.model_copy(update={"sessions": tuple(sessions)})
