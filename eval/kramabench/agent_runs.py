"""The given-files condition: the D3 analysis agent on one KramaBench task.

Only the task's labelled ``data_sources`` files (resolved as for the inlined
baseline) are mounted, read-only, at ``/data/<path under the visible root>``;
the prompt lists those container paths and sizes. Every model request,
repair attempts included, is a ledger step, so the D1 ``ReplayClient``
replays agent runs too; ``ReplayWorkspace`` returns the recorded cells and
rerun in place of the sandbox.

Scoring: the answer score and strict accuracy as for the baselines, and
verified success = strict and reproduced and observed access verified. A
run that does not submit has verified success False (this condition has a
program), not "not applicable".
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ds_research_agent.agent import (
    SYSTEM_PROMPT,
    TOOLS,
    ProgramEvent,
    Submission,
    Verification,
    Workspace,
    run_agent,
    user_prompt,
)
from ds_research_agent.ledger import (
    Answer,
    Batch,
    Ledger,
    Program,
    Run,
    Step,
    SubmissionRow,
    VerificationRow,
    canonical,
    utc_now,
)
from ds_research_agent.models import ChatMessage, ChatResult, ModelClient, ToolSpec
from ds_research_agent.models.ollama_client import to_ollama_tool
from ds_research_agent.sandbox import CellResult, InputMount, SandboxRun, SessionEnd
from eval.kramabench.baselines import (
    Harness,
    ReplayMismatch,
    request_record,
    score_run,
    sha256_text,
)
from eval.kramabench.tasks import Task, resolve_sources, resolved_files

CONDITION = "given_files"
ACCESS_POLICY = "labelled_files_mounted"

WorkspaceFor = Callable[[Task, int, Sequence[InputMount]], Workspace]


def prefix_sha256() -> str:
    """Identity of the cached prefix: system prompt and tool schemas."""
    return sha256_text(SYSTEM_PROMPT + canonical([to_ollama_tool(t) for t in TOOLS]))


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_inputs(
    h: Harness, task: Task
) -> tuple[list[InputMount], list[tuple[str, int]], dict[str, Any]]:
    """Mounts, the (container path, bytes) list for the prompt, and the manifest."""
    root = h.settings.kramabench.visible_root
    res = resolve_sources(task.domain, task.data_sources, root)
    mounts, listed, files = [], [], []
    for rel in resolved_files(res):
        host = root / rel
        container = f"/data/{rel}"
        size = host.stat().st_size
        mounts.append(InputMount(host_path=host, container_path=container))
        listed.append((container, size))
        files.append(
            {"path": container, "source": rel, "bytes": size, "sha256": _sha256_file(host)}
        )
    manifest = {
        "answer_type_visible": h.settings.eval.answer_type_visible,
        "resolution": [r.model_dump(mode="json") for r in res],
        "files": files,
    }
    return mounts, listed, manifest


class RecordingClient:
    """Records each request and response as a ledger step, in call order."""

    def __init__(self, inner: ModelClient, h: Harness, run_id: str) -> None:
        self.inner, self.h, self.run_id = inner, h, run_id
        self.seq = 0
        # A replay mismatch must fail the replay, not become a model error.
        self.fatal: ReplayMismatch | None = None

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        request = request_record(self.h.model, messages, tools)
        start = time.monotonic()
        result: ChatResult | None = None
        error: str | None = None
        try:
            result = await self.inner.chat(messages, tools)
            return result
        except ReplayMismatch as e:
            self.fatal = e
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            if self.fatal is None:
                self._record(request, result, error, time.monotonic() - start)

    def _record(
        self, request: dict[str, Any], result: ChatResult | None, error: str | None, wall: float
    ) -> None:
        u = result.usage if result else None
        self.h.ledger.add_step(
            Step(
                run_id=self.run_id,
                seq=self.seq,
                kind="model",
                request_sha256=sha256_text(canonical(request)),
                request=request,
                response=result.model_dump(mode="json") if result else None,
                prompt_eval_tokens=u.prompt_eval_tokens if u else None,
                prompt_eval_s=u.prompt_eval_ns / 1e9 if u and u.prompt_eval_ns else None,
                output_tokens=u.output_tokens if u else None,
                eval_s=u.eval_ns / 1e9 if u and u.eval_ns else None,
                load_s=u.load_ns / 1e9 if u and u.load_ns else None,
                thinking_chars=len(result.message.thinking or "") if result else None,
                wall_s=result.wall_s if result else wall,
                done_reason=result.done_reason if result else None,
                error=error,
            )
        )
        self.seq += 1


class LedgerRecorder:
    def __init__(self, ledger: Ledger, run_id: str) -> None:
        self.ledger, self.run_id = ledger, run_id
        self.seq = 0

    def program(self, e: ProgramEvent) -> None:
        rec: CellResult | SandboxRun | None = e.cell or e.run
        assert rec is not None
        self.ledger.add_program(
            Program(
                run_id=self.run_id,
                seq=e.seq,
                step=e.step,
                call_index=e.call_index,
                kind=e.kind,
                session=e.session,
                code=e.code,
                status=e.cell.status if e.cell else str(e.run.exit_code if e.run else None),
                wall_s=rec.wall_s,
                audit_complete=rec.audit.complete,
                observed_reads=[o.container_path for o in rec.observed_reads],
                replan_requested=e.replan_requested,
                record=rec.model_dump(mode="json"),
            )
        )
        self.seq = e.seq + 1

    def session_end(self, end: SessionEnd) -> None:
        """Kernel session audits, after the run's programs, for replay."""
        self.ledger.add_program(
            Program(
                run_id=self.run_id,
                seq=self.seq,
                step=-1,
                call_index=-1,
                kind="session_end",
                session=end.container,
                code="",
                status="complete" if end.audit.complete else "incomplete",
                wall_s=end.wall_s,
                audit_complete=end.audit.complete,
                observed_reads=[o.container_path for o in end.observed_reads],
                record=end.model_dump(mode="json"),
            )
        )
        self.seq += 1

    def submission(self, sub: Submission, step: int) -> None:
        self.ledger.put_submission(
            SubmissionRow(
                run_id=self.run_id,
                step=step,
                value=sub.answer,
                files_used=list(sub.files_used),
                program=sub.program,
                assumptions=list(sub.assumptions),
            )
        )

    def verification(self, v: Verification) -> None:
        self.ledger.put_verification(
            VerificationRow(
                run_id=self.run_id,
                comparator=v.comparator,
                rerun_exit_code=v.rerun_exit_code,
                rerun_answered=v.rerun_answered,
                rerun_answer=v.rerun_answer,
                reproduced=v.reproduced,
                audit_complete=v.audit_complete,
                observed_files=list(v.observed_files),
                claimed_files=list(v.claimed_files),
                access_verified=v.access_verified,
                detail=list(v.detail),
            )
        )


class ReplayWorkspace:
    """Returns one run's recorded cells, rerun, and session ends, refusing
    code that differs from what was recorded."""

    def __init__(self, programs: Sequence[Program]) -> None:
        # Setup cells (the comparator's) and model cells, in seq order.
        self._cells = [p for p in programs if p.kind in ("setup", "cell")]
        # Submit checks and the final rerun are workspace reruns, in seq order.
        self._reruns = [p for p in programs if p.kind in ("submit_check", "final_rerun")]
        self._ends = [
            SessionEnd.model_validate(p.record) for p in programs if p.kind == "session_end"
        ]
        self._session: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._session

    @staticmethod
    def _take(queue: list[Program], code: str, what: str) -> Program:
        if not queue:
            raise ReplayMismatch(f"more {what}s than recorded")
        p = queue.pop(0)
        if p.code != code:
            raise ReplayMismatch(f"{what} {p.seq}: code differs from the recording")
        return p

    async def execute(self, code: str, timeout_s: int) -> CellResult:
        p = self._take(self._cells, code, "cell")
        self._session = p.session
        return CellResult.model_validate(p.record)

    async def rerun(self, program: str) -> SandboxRun:
        return SandboxRun.model_validate(self._take(self._reruns, program, "rerun").record)

    async def close(self) -> list[SessionEnd]:
        return list(self._ends)


async def run_agent_task(
    h: Harness,
    batch: Batch,
    task: Task,
    repeat: int,
    client: ModelClient,
    workspace_for: WorkspaceFor,
    image_id: str | None,
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
    visible_at = task.answer_type if h.settings.eval.answer_type_visible else None
    out = await run_agent(
        rc,
        workspace_for(task, repeat, mounts),
        recorder,
        system=SYSTEM_PROMPT,
        user=user_prompt(task.query, listed, visible_at, h.settings.agent.max_steps),
        files=[c for c, _ in listed],
        settings=h.settings.agent,
        cell_timeout_s=h.settings.sandbox.cell_timeout_s,
    )
    if rc.fatal is not None:
        raise rc.fatal
    for end in out.sessions:
        recorder.session_end(end)
    sub = out.submission
    h.ledger.put_answer(
        Answer(
            run_id=run.run_id,
            answered=sub is not None,
            value=sub.answer if sub else None,
            parse_status="ok" if sub else out.stop_reason,
        )
    )
    score_run(h.ledger, run.run_id, task, CONDITION)
    h.ledger.finish_run(
        run.run_id,
        ended=utc_now(),
        wall_s=out.wall_s,
        stop_reason=out.stop_reason,
        error=out.error,
    )
    return run


def summary_extra(ledger: Ledger, run_ids: Sequence[str]) -> dict[str, Any]:
    """Agent-run counts for reports: steps, cells, and verification outcomes."""
    steps, cells, replans, checks, submitted, reproduced, access = [], [], 0, 0, 0, 0, 0
    for rid in run_ids:
        progs = ledger.programs(rid)
        steps.append(len(ledger.steps(rid)))
        cells.append(sum(p.kind == "cell" for p in progs))
        replans += sum(p.replan_requested for p in progs)
        checks += sum(p.kind == "submit_check" for p in progs)
        v = ledger.verification(rid)
        submitted += ledger.submission(rid) is not None
        reproduced += bool(v and v.reproduced)
        access += bool(v and v.access_verified)
    return {
        "model_requests_per_run": steps,
        "cells_per_run": cells,
        "replans_requested": replans,
        "submissions_returned_for_fix": checks,
        "submitted": submitted,
        "reproduced": reproduced,
        "access_verified": access,
    }
