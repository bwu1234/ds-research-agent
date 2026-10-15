"""The agentic-discovery condition (D2): the model searches and chooses files.

The run's answer is its file choice, not a value: ``choose_files`` IDs, with
member filters, expanded against the manifest (see
``ds_research_agent.agent.discovery``). Search is domain-filtered, as the
primary single-shot condition. Every model request and every search is a
ledger step (kinds ``model`` and ``search``), so a replay needs neither the
model nor the rag-toolkit server: ``ReplayClient`` takes the model steps and
``ReplaySearcher`` the search steps, each refusing a request that differs.

Scoring (profile ``discovery-v1``): against the task's resolved
``data_sources``, the chosen files' recall (``score``) and complete set
(``strict``); precision and counts are in the summary. Precision and extra
files are left out for a task with unresolved ``data_sources`` entries: an
unmatched entry (a typo, a directory) may name files the agent rightly chose. A run that chooses
nothing scores 0 and stays in the denominator; a task whose sources resolve
to no file is not scored. Cards are generated, so a run records the
catalogue generation it searched.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ds_research_agent.agent.discovery import (
    SYSTEM_PROMPT,
    TOOLS,
    CatalogueTools,
    Searcher,
    run_discovery,
    user_prompt,
)
from ds_research_agent.catalogue import Manifest
from ds_research_agent.ledger import Answer, Batch, Ledger, Run, Score, Step, canonical, utc_now
from ds_research_agent.models import ModelClient
from ds_research_agent.models.ollama_client import to_ollama_tool
from ds_research_agent.retrieval.mcp_adapter import SearchResponse
from eval.kramabench.agent_runs import RecordingClient
from eval.kramabench.baselines import Harness, ReplayMismatch, sha256_text
from eval.kramabench.discovery import gold_files, summarise
from eval.kramabench.tasks import Task

CONDITION = "discovery_agent"
ACCESS_POLICY = "catalogue_search_domain"
PROFILE = "discovery-v1"


def prefix_sha256() -> str:
    """The cached prompt prefix: system prompt and tool schemas."""
    tools = [to_ollama_tool(t) for t in TOOLS]
    return sha256_text(canonical({"system": SYSTEM_PROMPT, "tools": tools}))


def _search_request(
    query: str, corpus: str, top_k: int, max_chars: int, filters: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "query": query,
        "corpus": corpus,
        "top_k": top_k,
        "max_chars": max_chars,
        "filters": dict(filters) if filters else None,
    }


class RecordingSearcher:
    """Records each search as a ledger step, numbered with the model steps."""

    def __init__(self, inner: Searcher, h: Harness, run_id: str, rc: RecordingClient) -> None:
        self.inner, self.h, self.run_id, self.rc = inner, h, run_id, rc

    async def search(
        self,
        query: str,
        *,
        corpus: str,
        top_k: int,
        max_chars: int,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        request = _search_request(query, corpus, top_k, max_chars, filters)
        start = time.monotonic()
        resp: SearchResponse | None = None
        error: str | None = None
        try:
            resp = await self.inner.search(
                query, corpus=corpus, top_k=top_k, max_chars=max_chars, filters=filters
            )
            return resp
        except ReplayMismatch as e:
            self.rc.fatal = e
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            if self.rc.fatal is None:
                self.h.ledger.add_step(
                    Step(
                        run_id=self.run_id,
                        seq=self.rc.seq,
                        kind="search",
                        request_sha256=sha256_text(canonical(request)),
                        request=request,
                        response=resp.model_dump(mode="json") if resp else None,
                        wall_s=time.monotonic() - start,
                        error=error,
                    )
                )
                self.rc.seq += 1


class ReplaySearcher:
    """Returns a run's recorded search responses in order; a different request
    (query, corpus, filters, or limits) is a replay failure."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = [s for s in steps if s.kind == "search"]
        self._i = 0

    async def search(
        self,
        query: str,
        *,
        corpus: str,
        top_k: int,
        max_chars: int,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        if self._i >= len(self._steps):
            raise ReplayMismatch("more searches than recorded")
        step = self._steps[self._i]
        self._i += 1
        got = sha256_text(canonical(_search_request(query, corpus, top_k, max_chars, filters)))
        if got != step.request_sha256:
            raise ReplayMismatch(f"search step {step.seq}: request differs from the recording")
        if step.response is None:
            raise RuntimeError(step.error or "search failed (no response recorded)")
        return SearchResponse.model_validate(step.response)


def generation_id(index_dir: Path) -> str | None:
    gen = index_dir.parent / "generation.json"
    if not gen.exists():
        return None
    return str(json.loads(gen.read_text(encoding="utf-8"))["generation_id"])


async def run_discovery_task(
    h: Harness,
    batch: Batch,
    task: Task,
    repeat: int,
    client: ModelClient,
    searcher: Searcher,
    manifest: Manifest,
) -> Run:
    s = h.settings
    cards_digest = hashlib.sha256(
        canonical(
            [e.card_hash for e in manifest.entries] + [g.card_hash for g in manifest.groups]
        ).encode()
    ).hexdigest()
    run = Run(
        run_id=f"{batch.batch_id}/{task.key}/{repeat}",
        batch_id=batch.batch_id,
        task_key=task.key,
        parent_task_key=task.parent_key,
        split=h.split.split_of(task.key),
        condition=CONDITION,
        repeat=repeat,
        input_manifest={
            "corpus": manifest.corpus,
            "profiler_version": manifest.profiler_version,
            "entries": len(manifest.entries),
            "groups": len(manifest.groups),
            "cards_sha256": cards_digest,
            "domain_filter": task.domain,
            "collapse_groups": s.discovery.collapse_groups,
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
    out = await run_discovery(
        rc,
        tools,
        user=user_prompt(task.query, task.domain, s.discovery.max_steps),
        agent=s.agent,
        settings=s.discovery,
    )
    if rc.fatal is not None:
        raise rc.fatal
    chosen = out.chosen
    h.ledger.put_answer(
        Answer(
            run_id=run.run_id,
            answered=chosen is not None,
            value=(
                {
                    "items": [{"id": i, "members": list(m)} for i, m in chosen.items],
                    "files": list(chosen.files),
                    "reason": chosen.reason,
                    "searches": out.searches,
                    "cards_read": out.cards_read,
                    "rejected_choices": out.rejected_choices,
                    "discovered": len(out.discovered),
                }
                if chosen
                else {
                    "searches": out.searches,
                    "cards_read": out.cards_read,
                    "rejected_choices": out.rejected_choices,
                    "discovered": len(out.discovered),
                }
            ),
            parse_status="ok" if chosen else out.stop_reason,
        )
    )
    score_discovery(h.ledger, run.run_id, task, s.kramabench.visible_root)
    h.ledger.finish_run(
        run.run_id, ended=utc_now(), wall_s=out.wall_s, stop_reason=out.stop_reason, error=out.error
    )
    return run


def selection_metrics(chosen: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    want, got = set(gold), set(chosen)
    hit = want & got
    return {
        "recall": len(hit) / len(want),
        "complete": float(hit == want),
        "precision": len(hit) / len(got) if got else 0.0,
        "selected": float(len(got)),
        "excess": float(len(got - want)),
    }


def score_discovery(ledger: Ledger, run_id: str, task: Task, root: Path) -> Score | None:
    gold, _ = gold_files(task, root)
    if not gold:
        return None
    a = ledger.answer(run_id)
    files = a.value.get("files", []) if a and isinstance(a.value, dict) else []
    m = selection_metrics(files, gold)
    rec = Score(
        run_id=run_id,
        profile=PROFILE,
        answer_type="files",
        metric="file_recall",
        score=m["recall"],
        strict=bool(m["complete"]),
        verified_success=None,
        scored=utc_now(),
    )
    ledger.put_score(rec)
    return rec


def rows(
    ledger: Ledger, batch: Batch, tasks: Mapping[str, Task], root: Path
) -> list[dict[str, Any]]:
    """One row per task (repeat 0), in the format of ``eval.kramabench.discovery``."""
    out = []
    by_key = {r.task_key: r for r in ledger.runs(batch.batch_id) if r.repeat == 0}
    for key in batch.task_keys:
        t = tasks[key]
        gold, unresolved = gold_files(t, root)
        row: dict[str, Any] = {"task": key, "domain": t.domain, "gold": len(gold)}
        row["unresolved"] = unresolved
        r = by_key.get(key)
        if not gold or r is None:
            out.append(row)
            continue
        a = ledger.answer(r.run_id)
        v = a.value if a and isinstance(a.value, dict) else {}
        steps = ledger.steps(r.run_id)
        m = selection_metrics(v.get("files", []), gold)
        if unresolved:
            # Part of the gold set is unknown, so extra files may be gold.
            del m["precision"], m["excess"]
        row |= {
            "latency_s": float(r.wall_s or 0.0),
            **m,
            "chose": float(bool(a and a.answered)),
            "searches": float(v.get("searches", 0)),
            "cards_read": float(v.get("cards_read", 0)),
            "model_requests": float(sum(s.kind == "model" for s in steps)),
            "wall_s": float(r.wall_s or 0.0),
        }
        out.append(row)
    return out


def summary(ledger: Ledger, batch: Batch, tasks: Mapping[str, Task], root: Path) -> dict[str, Any]:
    rs = rows(ledger, batch, tasks, root)
    out: dict[str, Any] = {
        "ranker": f"{CONDITION} ({batch.batch_id})",
        "scope": "domain",
        "tasks": rs,
    }
    out["summary"] = summarise(rs)
    walls = [r["wall_s"] for r in rs if r.get("gold") and "wall_s" in r]
    out["summary"]["wall_s"] = {
        "median": statistics.median(walls) if walls else None,
        "total": round(sum(walls), 1),
    }
    out["summary"]["stop_reasons"] = dict(
        sorted(
            {
                reason: sum(1 for r in ledger.runs(batch.batch_id) if r.stop_reason == reason)
                for reason in {r.stop_reason for r in ledger.runs(batch.batch_id)}
                if reason is not None
            }.items()
        )
    )
    return out
