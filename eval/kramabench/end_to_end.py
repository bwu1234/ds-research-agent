"""The end-to-end condition (D4): single-shot discovery, then the analysis loop.

Per task: one domain-filtered ``rag_search`` on the question, collapsed into
groups, with the first ``end_to_end.seed_cards`` results listed in the
prompt as the run's discovered set (``ds_research_agent.agent.end_to_end``).
The agent selects discovered IDs in ``run_python``; only selected files are
mounted, at ``/data/<catalogue path>``. The analysis budget is ``agent``'s,
as for ``given_files``; the seed search is the discovery step, timed apart.

Ledger steps: ``search`` (the seed search), ``model`` (every request), and
``selection`` (each ``run_python`` selection as the application resolved it:
items, problems, and the newly mounted files with their hashes), numbered in
call order. Each kernel session and rerun records its own mounts
(``inputs``). A replay needs neither the model, the rag-toolkit server, nor
Docker: the recorded search, responses, cells, and reruns are returned.
Selections are resolved again from the manifest and their files hashed
again, so like a ``given_files`` replay (which hashes its inputs) it needs
the data store and catalogue the batch ran on.

A seed search that finds nothing leaves nothing to select; the prompt says
so, and the report counts such runs (``empty_seed``).

Scoring is ``given_files``': answer score, strict accuracy, and verified
success (strict, reproduced, and observed access verified). The selection is
also scored against the task's resolved ``data_sources`` (as
``discovery-v1``, in the report only), next to whether the seed results made
the gold set reachable at all.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ds_research_agent.agent import run_agent
from ds_research_agent.agent.discovery import CatalogueTools, Searcher
from ds_research_agent.agent.end_to_end import (
    SYSTEM_PROMPT,
    TOOLS,
    CatalogueAccess,
    user_prompt,
)
from ds_research_agent.agent.workspace import SelectableWorkspace
from ds_research_agent.catalogue import Manifest
from ds_research_agent.ledger import Answer, Batch, Ledger, Run, Step, canonical, utc_now
from ds_research_agent.models import ModelClient
from ds_research_agent.models.ollama_client import to_ollama_tool
from eval.kramabench.agent_runs import LedgerRecorder, RecordingClient
from eval.kramabench.baselines import Harness, score_run, sha256_text
from eval.kramabench.discovery import collapse, collapse_map, document_files, gold_files
from eval.kramabench.discovery_agent import RecordingSearcher, generation_id, selection_metrics
from eval.kramabench.tasks import Task

CONDITION = "end_to_end"
ACCESS_POLICY = "catalogue_selected_domain"

WorkspaceFor = Callable[[Task, int], SelectableWorkspace]


def prefix_sha256() -> str:
    """Identity of the cached prefix, in the form ``given_files`` uses."""
    return sha256_text(SYSTEM_PROMPT + canonical([to_ollama_tool(t) for t in TOOLS]))


class SelectionRecorder:
    """Writes each selection as a ``selection`` step, numbered with the rest."""

    def __init__(self, h: Harness, run_id: str, rc: RecordingClient) -> None:
        self.h, self.run_id, self.rc = h, run_id, rc

    def __call__(self, record: dict[str, Any]) -> None:
        request = {"items": record["items"]}
        self.h.ledger.add_step(
            Step(
                run_id=self.run_id,
                seq=self.rc.seq,
                kind="selection",
                request_sha256=sha256_text(canonical(request)),
                request=request,
                response=record,
            )
        )
        self.rc.seq += 1


def cards_sha256(manifest: Manifest) -> str:
    return hashlib.sha256(
        canonical(
            [e.card_hash for e in manifest.entries] + [g.card_hash for g in manifest.groups]
        ).encode()
    ).hexdigest()


async def run_end_to_end_task(
    h: Harness,
    batch: Batch,
    task: Task,
    repeat: int,
    client: ModelClient,
    workspace_for: WorkspaceFor,
    image_id: str | None,
    searcher: Searcher,
    manifest: Manifest,
) -> Run:
    s = h.settings
    e2e = s.end_to_end
    if e2e is None:
        raise ValueError("the end_to_end condition needs the end_to_end settings section")
    start = time.monotonic()
    run = Run(
        run_id=f"{batch.batch_id}/{task.key}/{repeat}",
        batch_id=batch.batch_id,
        task_key=task.key,
        parent_task_key=task.parent_key,
        split=h.split.split_of(task.key),
        condition=CONDITION,
        repeat=repeat,
        input_manifest={
            "answer_type_visible": s.eval.answer_type_visible,
            "sandbox_image": image_id,
            "corpus": manifest.corpus,
            "profiler_version": manifest.profiler_version,
            "entries": len(manifest.entries),
            "groups": len(manifest.groups),
            "cards_sha256": cards_sha256(manifest),
            "domain_filter": task.domain,
            "collapse_groups": s.discovery.collapse_groups,
            "seed_cards": e2e.seed_cards,
        },
        access_policy=ACCESS_POLICY,
        catalogue_generation=generation_id(s.rag_service.index_dir),
        started=utc_now(),
        repo_revision=h.repo_revision,
    )
    h.ledger.add_run(run)
    rc = RecordingClient(client, h, run.run_id)
    tools = CatalogueTools(
        RecordingSearcher(searcher, h, run.run_id, rc),
        manifest,
        s.catalogue.cards_dir,
        s.catalogue.corpus,
        {"equals": {"domain": task.domain}},
        s.discovery,
    )
    results = await tools.search(task.query, limit=e2e.seed_cards)
    if rc.fatal is not None:
        raise rc.fatal
    workspace = workspace_for(task, repeat)
    access = CatalogueAccess(
        tools, workspace, s.catalogue.data_root, e2e, SelectionRecorder(h, run.run_id, rc)
    )
    recorder = LedgerRecorder(h.ledger, run.run_id)
    visible_at = task.answer_type if s.eval.answer_type_visible else None
    out = await run_agent(
        rc,
        workspace,
        recorder,
        system=SYSTEM_PROMPT,
        user=user_prompt(
            task.query,
            task.domain,
            results if tools.discovered.ids else None,
            visible_at,
            s.agent.max_steps,
        ),
        files=[],
        settings=s.agent,
        cell_timeout_s=s.sandbox.cell_timeout_s,
        tools=TOOLS,
        access=access,
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
    # The run's wall clock includes the seed search; its own time is the
    # search step's.
    h.ledger.finish_run(
        run.run_id,
        ended=utc_now(),
        wall_s=round(time.monotonic() - start, 3),
        stop_reason=out.stop_reason,
        error=out.error,
    )
    return run


def selected_files(ledger: Ledger, run_id: str) -> list[str]:
    """Store-relative paths of every file the run selected, in selection order."""
    out: list[str] = []
    for st in ledger.steps(run_id):
        if st.kind == "selection" and st.response:
            out += [a["path"].removeprefix("/data/") for a in st.response["added"]]
    return out


def seed_files(
    ledger: Ledger, run_id: str, manifest: Manifest, k: int, collapse_groups: bool
) -> list[str]:
    """Files the seed search made selectable: the members of its first k
    distinct results (collapsed if the batch collapsed them), scored as the
    single-shot baseline's @k."""
    search = next((st for st in ledger.steps(run_id) if st.kind == "search"), None)
    if search is None or search.response is None:
        return []
    ranked = list(dict.fromkeys(r["document_id"] for r in search.response["results"]))
    if collapse_groups:
        ranked = collapse(ranked, collapse_map(manifest))
    members = document_files(manifest)
    return [f for d in ranked[:k] for f in members[d]]


def summary_extra(
    ledger: Ledger,
    batch: Batch,
    tasks: Mapping[str, Task],
    root: Path,
    manifest: Manifest | None,
) -> dict[str, Any]:
    """Discovery inside end-to-end runs: seed search time, the selection's
    recall and complete set against resolved ``data_sources`` (and, with the
    run's own catalogue, whether the seed results made the gold set
    reachable at all), selections, and kernel restarts for new mounts.

    ``seed_complete`` is left out for a run whose recorded ``cards_sha256``
    differs from ``manifest`` (counted as ``manifest_mismatch``): a rebuilt
    catalogue would map the recorded results to other files."""
    k = int((batch.config.get("end_to_end") or {}).get("seed_cards", 0))
    collapse_groups = bool((batch.config.get("discovery") or {}).get("collapse_groups", True))
    cards = cards_sha256(manifest) if manifest is not None else None
    rows: list[dict[str, Any]] = []
    mismatched = 0
    for r in ledger.runs(batch.batch_id):
        gold, unresolved = gold_files(tasks[r.task_key], root)
        steps = ledger.steps(r.run_id)
        sel = [st for st in steps if st.kind == "selection"]
        search = next((st for st in steps if st.kind == "search"), None)
        sessions = [p for p in ledger.programs(r.run_id) if p.kind == "session_end"]
        row: dict[str, Any] = {
            "search_s": search.wall_s if search else None,
            "empty_seed": bool(search and search.response and not search.response["results"]),
            "selections": len(sel),
            "refused_selections": sum(bool(st.response and st.response["problems"]) for st in sel),
            "kernel_sessions": len(sessions),
        }
        same_catalogue = manifest is not None and r.input_manifest.get("cards_sha256") == cards
        if manifest is not None and not same_catalogue:
            mismatched += 1
        if gold:
            m = selection_metrics(selected_files(ledger, r.run_id), gold)
            if unresolved:
                del m["precision"], m["excess"]
            row |= m
            if manifest is not None and same_catalogue:
                reach = set(seed_files(ledger, r.run_id, manifest, k, collapse_groups))
                row["seed_complete"] = float(set(gold) <= reach)
        rows.append(row)

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    searches = [r["search_s"] for r in rows if r["search_s"] is not None]
    return {
        "scored_tasks": sum("recall" in r for r in rows),
        "selection_recall": mean("recall"),
        "selection_complete": mean("complete"),
        "selection_precision": mean("precision"),
        "seed_complete": mean("seed_complete"),
        "manifest_mismatch": mismatched,
        "empty_seed": sum(r["empty_seed"] for r in rows),
        "selections_per_run": mean("selections"),
        "refused_selections": sum(r["refused_selections"] for r in rows),
        # A run's first session starts with its first cell; each later one
        # follows a kernel death or a selection that added files.
        "kernel_sessions_per_run": mean("kernel_sessions"),
        "search_s": {
            "median": round(statistics.median(searches), 3) if searches else None,
            "total": round(sum(searches), 1),
        },
    }
