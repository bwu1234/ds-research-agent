"""Retrieval-only discovery scoring (offline; no index service, no model)."""

from __future__ import annotations

import asyncio

from eval.kramabench.discovery import (
    Bm25Ranker,
    bootstrap_mean,
    summarise,
    task_metrics,
    tokens,
)

MEMBERS = {"a.md": ("a",), "b.md": ("b",), "c.md": ("c",), "g.md": ("x", "y")}


def test_recall_hit_and_complete_count_distinct_documents() -> None:
    m = task_metrics(["c.md", "a.md", "b.md"], MEMBERS, ["a", "b"])
    assert m["returned"] == 3
    assert (m["recall@1"], m["hit@1"], m["complete@1"]) == (0.0, 0.0, 0.0)
    assert (m["recall@5"], m["hit@5"], m["complete@5"]) == (1.0, 1.0, 1.0)


def test_a_document_standing_for_several_files_counts_each() -> None:
    m = task_metrics(["g.md"], MEMBERS, ["x", "y", "z"])
    assert m["recall@1"] == 2 / 3 and m["complete@1"] == 0.0


def test_unknown_documents_contribute_nothing() -> None:
    assert task_metrics(["zzz.md"], MEMBERS, ["a"])["hit@20"] == 0.0


def test_tokens_split_identifiers() -> None:
    assert tokens("State_MSA-Fraud/2024.csv") == ["state", "msa", "fraud", "2024", "csv"]


def test_bm25_ranks_the_matching_card_first_and_filters_by_domain() -> None:
    docs = {
        "fraud.md": "legal fraud reports by state and metropolitan area",
        "theft.md": "legal identity theft reports by state",
        "fire.md": "wildfire acres burned by state",
    }
    domains = {"fraud.md": "legal", "theft.md": "legal", "fire.md": "wildfire"}
    r = Bm25Ranker(docs, domains)
    assert asyncio.run(r.rank("fraud reports", None))[0] == "fraud.md"
    assert asyncio.run(r.rank("state", "wildfire")) == ["fire.md"]
    assert asyncio.run(r.rank("nothing matches", None)) == []


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    vals = {"t1": 0.0, "t2": 1.0, "t3": 1.0, "t4": 0.5}
    strata = {"t1": "a", "t2": "a", "t3": "b", "t4": "b"}
    first = bootstrap_mean(vals, strata, resamples=500, seed=1)
    assert first == bootstrap_mean(vals, strata, resamples=500, seed=1)
    mean, lo, hi = first
    assert mean == 0.625 and lo <= mean <= hi


def test_summary_excludes_tasks_without_gold_files_but_lists_them() -> None:
    base = {f"{m}@{k}": 1.0 for m in ("recall", "hit", "complete") for k in (1, 5, 10, 20)}
    rows = [
        {"task": "t1", "domain": "a", "gold": 1, "unresolved": 0, "latency_s": 0.1,
         "returned": 1.0, **base},
        {"task": "t2", "domain": "a", "gold": 0, "unresolved": 1},
    ]  # fmt: skip
    s = summarise(rows)
    assert s["pooled"]["tasks"] == 1 and s["unscored_tasks"] == ["t2"]
    assert s["unresolved_entries"] == 1
