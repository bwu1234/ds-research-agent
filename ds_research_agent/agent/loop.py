"""The given-files analysis loop: model step, run code, repeat, submit, rerun.

Each step is one model turn through the bounded tool-call repair policy;
the conversation is append-only, so Ollama can reuse the cached prefix.
``run_python`` cells go to the workspace's persistent kernel and come back
as bounded text. ``submit_answer`` is checked, then its program is run in a
fresh sandbox. A program that fails or prints no answer line is returned to
the model for a fix (at most ``submit_checks`` times, each costing a step);
otherwise that run is the final rerun, compared with ``reproduction-v1`` and
the read audit. Only whether the program runs is fed back: a reproduction or
access mismatch is never revealed, so it stays final. Budgets (steps, wall
clock) end the run with a stop reason and no answer; the loop never guesses.

The loop never sees gold answers: scoring is the evaluator's job. Model
requests are recorded by the client the caller passes in (the harness wraps
its client); programs, the submission, and the verification go to the
``Recorder``.
"""

from __future__ import annotations

import asyncio
import difflib
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
    NO_PROGRESS,
    NO_TOOL_CALL,
    REPLAN,
    SUBMIT_FAILED,
    TOOLS,
    TRUNCATED,
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
    """One exploration cell, a submitted program that failed its run and was
    returned (``submit_check``), or the final program's fresh rerun."""

    seq: int
    step: int  # the model step that called it; for a rerun, the submit step
    call_index: int
    kind: Literal["cell", "submit_check", "final_rerun"]
    session: str | None
    code: str
    cell: CellResult | None = None
    run: SandboxRun | None = None
    # The tool result asked for a change of approach: after a repeated error,
    # or after repeated near-identical successful cells (no progress).
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
    note = TRUNCATED.format(part="first", limit=len(text))
    return text + ("\n" + note if truncated else "")


def _tail(text: str, limit: int, truncated: bool) -> str:
    if len(text) > limit:
        text, truncated = text[-limit:], True
    note = TRUNCATED.format(part="last", limit=len(text))
    return (note + "\n" if truncated else "") + text


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


def run_problem(run: SandboxRun) -> str | None:
    """Why a submitted program's run gives no answer; None when it prints one."""
    if run.timed_out:
        return "timed out"
    if run.exit_code != 0:
        return f"exited with code {run.exit_code}"
    if not program_answer(run.stdout)[0]:
        return 'printed no {"answer": ...} JSON as its last line'
    return None


def render_run_output(run: SandboxRun, limit: int) -> str:
    """The failed program's output tails, where the traceback or last line is."""
    parts = []
    if run.stdout or run.stdout_truncated:
        parts.append("stdout:\n" + _tail(run.stdout.rstrip("\n"), limit, False))
    if run.stderr or run.stderr_truncated:
        parts.append("stderr:\n" + _tail(run.stderr.rstrip("\n"), limit, False))
    return "\n".join(parts) or "(no output)"


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
        self.checks = 0
        self.last_code: str | None = None
        self.same_code = 0

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
        stuck = self.no_progress(code, cell)
        content = render_cell(cell, self.s.max_tool_output_chars)
        if not cell.kernel_alive:
            content += "\n" + KERNEL_RESTARTED.format(why=cell.status)
        if replan:
            content += "\n" + REPLAN.format(n=self.repeats)
        elif stuck:
            content += "\n" + NO_PROGRESS.format(n=self.same_code)
        replan = replan or stuck
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

    def no_progress(self, code: str, cell: CellResult) -> bool:
        """Count successful cells nearly identical to the one before; True from
        ``no_progress_after_repeats`` such repeats in a row. Errors are the
        re-plan rule's, so an error resets the count."""
        prev, self.last_code = self.last_code, code
        similar = (
            cell.status == "ok"
            and prev is not None
            and difflib.SequenceMatcher(None, prev, code).ratio() >= self.s.no_progress_similarity
        )
        self.same_code = self.same_code + 1 if similar else 0
        n = self.s.no_progress_after_repeats
        return n > 0 and self.same_code >= n

    async def rerun(self, program: str) -> tuple[SandboxRun | None, str | None]:
        try:
            return await self.workspace.rerun(program), None
        except SandboxUnavailable:
            raise
        except SandboxError as e:
            return None, f"rerun failed in the sandbox: {e}"

    async def try_submit(
        self, sub: Submission, step: int, index: int
    ) -> tuple[SandboxRun | None, str | None] | ChatMessage:
        """Run the submitted program; a tool reply when it is returned for a fix."""
        run, error = await self.rerun(sub.program)
        problem = run_problem(run) if run is not None else None
        if (
            run is None
            or problem is None
            or self.checks >= self.s.submit_checks
            or step >= self.s.max_steps  # no turn left to fix it: the run is final
        ):
            return run, error
        self.checks += 1
        self.recorder.program(
            ProgramEvent(
                seq=self.seq,
                step=step,
                call_index=index,
                kind="submit_check",
                session=run.container,
                code=sub.program,
                run=run,
            )
        )
        self.seq += 1
        output = render_run_output(run, self.s.max_tool_output_chars)
        return _tool("submit_answer", SUBMIT_FAILED.format(problem=problem, output=output))

    def verify(
        self, sub: Submission, step: int, index: int, run: SandboxRun | None, error: str | None
    ) -> Verification:
        detail = [error] if error else []
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
            submitted: tuple[Submission, int, SandboxRun | None, str | None] | None = None
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
                        tried = await self.try_submit(sub, steps, i)
                        if isinstance(tried, ChatMessage):
                            replies.append(tried)
                        else:
                            submitted = (sub, i, *tried)
            except SandboxUnavailable:
                raise  # infrastructure, not the run: the batch stops, resumable
            except SandboxError as e:
                return self.outcome("sandbox_error", steps, str(e))
            messages += replies
            if submitted is None:
                self.budget_notice(messages, steps)
            else:
                sub, index, run, error = submitted
                self.recorder.submission(sub, steps)
                verification = self.verify(sub, steps, index, run, error)
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
