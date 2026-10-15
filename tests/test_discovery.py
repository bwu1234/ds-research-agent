"""Retrieval-only discovery scoring (offline; no index service, no model)."""

from __future__ import annotations

import asyncio

from eval.kramabench.discovery import (
    Bm25Ranker,
    bootstrap_mean,
    collapse,
    collapse_map,
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


def test_file_metrics_ignore_group_cards_and_groups_are_counted() -> None:
    m = task_metrics(["g.md", "a.md"], MEMBERS, ["x", "y", "a"], frozenset({"g.md"}))
    assert m["recall@1"] == 2 / 3 and m["file_recall@1"] == 0.0
    assert m["complete@5"] == 1.0 and m["file_recall@5"] == 1 / 3
    assert m["groups@1"] == 1.0 and m["groups@5"] == 1.0
    assert "file_recall@1" not in task_metrics(["a.md"], MEMBERS, ["a"])


def test_paired_differences_use_tasks_scored_in_both() -> None:
    from eval.kramabench.discovery_paired import paired

    a = [
        {"task": "t1", "domain": "d", "gold": 2, "recall@5": 1.0},
        {"task": "t2", "domain": "d", "gold": 1, "recall@5": 0.5},
        {"task": "t3", "domain": "d", "gold": 0},
    ]
    b = [
        {"task": "t1", "domain": "d", "gold": 2, "recall@5": 0.5},
        {"task": "t2", "domain": "d", "gold": 1, "recall@5": 0.5},
    ]
    out = paired(a, b, ["recall@5", "complete@10"])
    assert out["tasks"] == 2 and out["recall@5"][0] == 0.25
    assert "complete@10" not in out


def test_collapse_shows_a_grouped_file_as_its_group_once() -> None:
    to_group = {"a.md": "g.md", "b.md": "g.md"}
    assert collapse(["c.md", "b.md", "g.md", "a.md", "d.md"], to_group) == ["c.md", "g.md", "d.md"]


def test_collapse_map_prefers_the_largest_group() -> None:
    from ds_research_agent.catalogue import CatalogueEntry, CatalogueGroup, Manifest

    def e(p: str) -> CatalogueEntry:
        return CatalogueEntry(
            catalogue_id=p,
            domain="d",
            file_path=p,
            size=1,
            file_hash="sha256:0",
            format="csv",
            document_id=f"{p}.md",
            card_hash="sha256:0",
            profiler_version=3,
            parse_status="ok",
        )

    def g(gid: str, members: tuple[str, ...]) -> CatalogueGroup:
        return CatalogueGroup(
            group_id=gid,
            domain="d",
            rules=("pattern",),
            member_ids=members,
            directory="d",
            format="csv",
            document_id=f"{gid}.md",
            card_hash="sha256:0",
            profiler_version=3,
        )

    m = Manifest(
        profiler_version=3,
        corpus="c",
        entries=[e("x"), e("y"), e("z")],
        skipped=[],
        groups=[g("small", ("x", "y")), g("big", ("x", "y", "z"))],
    )
    assert collapse_map(m) == {"x.md": "big.md", "y.md": "big.md", "z.md": "big.md"}


def test_summary_averages_each_metric_over_the_rows_that_have_it() -> None:
    rows = [
        {"task": "a", "domain": "d", "gold": 1, "unresolved": 0, "latency_s": 1.0,
         "recall": 1.0, "precision": 0.5},
        {"task": "b", "domain": "d", "gold": 1, "unresolved": 1, "latency_s": 1.0,
         "recall": 0.0},
    ]  # fmt: skip
    s = summarise(rows)["pooled"]
    assert s["recall"][0] == 0.5 and s["precision"][0] == 0.5
    assert s["precision_tasks"] == 1 and "recall_tasks" not in s


def test_paired_metrics_can_have_different_names() -> None:
    from eval.kramabench.discovery_paired import paired

    a = [{"task": "t", "domain": "d", "gold": 1, "complete": 1.0}]
    b = [{"task": "t", "domain": "d", "gold": 1, "complete@10": 0.0}]
    assert paired(a, b, ["complete:complete@10"])["complete:complete@10"][0] == 1.0
