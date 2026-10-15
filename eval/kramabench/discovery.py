"""Retrieval-only discovery scores (D2): no generative model is called.

For each development task, the task's question is the query, and the gold
set is the task's ``data_sources`` resolved to files in the agent-visible
store (``data_sources`` entries that resolve to nothing are counted, not
scored: no ranker can return a file that is not in the store). A ranker
returns catalogue documents in rank order; each document maps to the files it
stands for (one, for a file card). Metrics at k count the first k distinct
documents:

- ``recall@k``: the share of a task's gold files among them;
- ``hit@k``: at least one gold file is among them;
- ``complete@k``: every gold file is among them.

Scope ``pooled`` searches the whole catalogue; ``domain`` adds the task's
domain as a filter, as a per-domain condition would. Uncertainty is a
percentile bootstrap over tasks, stratified by domain. ``data_sources`` and
the results are evaluator-side material: the output stays local.

    python -m eval.kramabench.discovery --config config/local.kramabench.yaml \
        --ranker rag --scope pooled --out data/measurements/discovery-rag.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from ds_research_agent.catalogue.models import Manifest, load_manifest
from ds_research_agent.config import McpSettings, Settings, load_settings
from ds_research_agent.retrieval.mcp_adapter import McpRetrieval
from eval.kramabench.split import frozen_split
from eval.kramabench.tasks import Task, resolve_sources, resolved_files

KS = (1, 5, 10, 20)
# rag_search returns at most 20 passages; several can come from one card.
MAX_PASSAGES = 20
Scope = Literal["pooled", "domain"]


class Ranker(Protocol):
    name: str

    async def rank(self, query: str, domain: str | None) -> list[str]:
        """Catalogue document IDs, best first, without duplicates."""
        ...


def gold_files(task: Task, root: Path) -> tuple[tuple[str, ...], int]:
    """The task's gold files (store-relative paths) and its unresolved entries."""
    res = resolve_sources(task.domain, task.data_sources, root)
    unresolved = sum(1 for r in res if not r.files)
    return resolved_files(res), unresolved


def document_files(manifest: Manifest) -> dict[str, tuple[str, ...]]:
    """Document ID -> the files it stands for."""
    return {e.document_id: (e.file_path,) for e in manifest.entries}


def task_metrics(
    ranked: Sequence[str], members: Mapping[str, tuple[str, ...]], gold: Sequence[str]
) -> dict[str, float]:
    want = set(gold)
    out: dict[str, float] = {"returned": float(len(ranked))}
    for k in KS:
        found = {f for d in ranked[:k] for f in members.get(d, ())} & want
        out[f"recall@{k}"] = len(found) / len(want)
        out[f"hit@{k}"] = float(bool(found))
        out[f"complete@{k}"] = float(found == want)
    return out


def bootstrap_mean(
    values: Mapping[str, float],
    strata: Mapping[str, str],
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, resampling tasks within each stratum."""
    keys = sorted(values)
    by: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by[strata[k]].append(k)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [rng.choice(ks) for ks in by.values() for _ in ks]
        means.append(sum(values[k] for k in sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(math.ceil(0.975 * (resamples - 1)))]
    return sum(values.values()) / len(values), lo, hi


# --- rankers ------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-case alphanumeric runs; ``_``, ``-``, ``/`` and ``.`` split words."""
    return _TOKEN.findall(text.lower())


class Bm25Ranker:
    """Okapi BM25 over whole cards, one document per card.

    A document-level lexical reference that needs no index service and no
    chunking; it is a baseline, not rag-toolkit's BM25 stage.
    """

    name = "bm25"

    def __init__(
        self,
        docs: Mapping[str, str],
        domains: Mapping[str, str],
        k1: float = 1.2,
        b: float = 0.75,
        top: int = MAX_PASSAGES,
    ) -> None:
        self._ids = sorted(docs)
        self._domains = domains
        self._tf = [Counter(tokens(docs[d])) for d in self._ids]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._ids)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._k1, self._b, self._top = k1, b, top

    async def rank(self, query: str, domain: str | None) -> list[str]:
        q = set(tokens(query))
        scored = []
        for i, d in enumerate(self._ids):
            if domain is not None and self._domains[d] != domain:
                continue
            tf, norm = self._tf[i], 1 - self._b + self._b * self._len[i] / self._avg
            s = sum(
                self._idf[t] * tf[t] * (self._k1 + 1) / (tf[t] + self._k1 * norm)
                for t in q
                if t in tf
            )
            if s > 0:
                scored.append((-s, d))
        scored.sort()
        return [d for _, d in scored[: self._top]]


class RagRanker:
    """rag-toolkit's ``rag_search`` over MCP, the contract the agent will use."""

    def __init__(self, retrieval: McpRetrieval, corpus: str, name: str) -> None:
        self._r, self._corpus, self.name = retrieval, corpus, name

    async def rank(self, query: str, domain: str | None) -> list[str]:
        filters = {"equals": {"domain": domain}} if domain else None
        resp = await self._r.search(
            query, corpus=self._corpus, top_k=MAX_PASSAGES, max_chars=200, filters=filters
        )
        out: list[str] = []
        for r in resp.results:
            if r.document_id not in out:
                out.append(r.document_id)
        return out


def mode_variant(settings: Settings, mode: str) -> McpSettings:
    """MCP settings serving the generated rag-toolkit config with another
    ``retrieval.mode``. Query-time only: the index is unchanged."""
    gen = settings.rag_service.generated_config
    cfg = yaml.safe_load(gen.read_text(encoding="utf-8"))
    cfg["retrieval"]["mode"] = mode
    out = gen.with_name(f"{gen.stem}.{mode}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True), encoding="utf-8")
    args = [str(out) if a == str(gen) else a for a in settings.mcp.args]
    if args == settings.mcp.args:
        raise ValueError("mcp.args does not name rag_service.generated_config")
    return settings.mcp.model_copy(update={"args": args})


# --- evaluation ---------------------------------------------------------------


async def evaluate(
    ranker: Ranker,
    tasks: Sequence[Task],
    root: Path,
    members: Mapping[str, tuple[str, ...]],
    scope: Scope,
) -> dict[str, Any]:
    rows = []
    for t in tasks:
        gold, unresolved = gold_files(t, root)
        if not gold:
            rows.append({"task": t.key, "domain": t.domain, "gold": 0, "unresolved": unresolved})
            continue
        start = time.monotonic()
        ranked = await ranker.rank(t.query, t.domain if scope == "domain" else None)
        latency = time.monotonic() - start
        rows.append(
            {
                "task": t.key,
                "domain": t.domain,
                "gold": len(gold),
                "unresolved": unresolved,
                "latency_s": round(latency, 3),
                **task_metrics(ranked, members, gold),
            }
        )
    return {"ranker": ranker.name, "scope": scope, "tasks": rows, "summary": summarise(rows)}


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["gold"]]
    strata = {r["task"]: r["domain"] for r in scored}
    metrics = ["returned"] + [f"{m}@{k}" for m in ("recall", "hit", "complete") for k in KS]

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"tasks": len(subset)}
        for m in metrics:
            vals = {r["task"]: float(r[m]) for r in subset}
            mean, lo, hi = bootstrap_mean(vals, strata)
            out[m] = [round(mean, 3), round(lo, 3), round(hi, 3)]
        return out

    lat = sorted(r["latency_s"] for r in scored)
    by_domain = defaultdict(list)
    for r in scored:
        by_domain[r["domain"]].append(r)
    return {
        "pooled": block(scored),
        "by_domain": {d: block(rs) for d, rs in sorted(by_domain.items())},
        "unscored_tasks": [r["task"] for r in rows if not r["gold"]],
        "unresolved_entries": sum(r["unresolved"] for r in rows),
        "latency_s": {
            "first": scored[0]["latency_s"] if scored else None,
            "median": lat[len(lat) // 2] if lat else None,
            "max": lat[-1] if lat else None,
        },
        "uncertainty": "mean [2.5%, 97.5%], percentile bootstrap of tasks within domain",
    }


def _print(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    print(f"{result['ranker']} ({result['scope']}): {s['pooled']['tasks']} tasks")
    cols = ["recall@5", "recall@20", "hit@5", "complete@20", "returned"]
    print(f"  {'':12s}" + "".join(f"{c:>20s}" for c in cols))
    for name, b in [("pooled", s["pooled"]), *s["by_domain"].items()]:
        cells = "".join(f"{b[c][0]:>8.3f} [{b[c][1]:.2f},{b[c][2]:.2f}]" for c in cols)
        print(f"  {name:12s}{cells}  n={b['tasks']}")
    print(f"  latency s: {s['latency_s']}; unresolved data_sources: {s['unresolved_entries']}")


async def _main(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    manifest = load_manifest(settings.catalogue.manifest_path)
    split, all_tasks = frozen_split(settings)
    tasks = [all_tasks[k] for k in split.dev]
    root = settings.kramabench.visible_root
    members = document_files(manifest)
    if args.ranker == "bm25":
        cards = settings.catalogue.cards_dir
        docs = {e.document_id: (cards / e.document_id).read_text("utf-8") for e in manifest.entries}
        domains = {e.document_id: e.domain for e in manifest.entries}
        result = await evaluate(Bm25Ranker(docs, domains), tasks, root, members, args.scope)
    else:
        mcp = mode_variant(settings, args.mode) if args.mode else settings.mcp
        name = f"rag-{args.mode or 'configured'}"
        async with McpRetrieval(mcp, errlog=open(args.out.with_suffix(".server.log"), "w")) as r:
            result = await evaluate(
                RagRanker(r, settings.catalogue.corpus, name), tasks, root, members, args.scope
            )
            result["summary"]["startup_s"] = round(r.startup_s or 0.0, 2)
    result["catalogue"] = {
        "corpus": manifest.corpus,
        "profiler_version": manifest.profiler_version,
        "entries": len(manifest.entries),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    _print(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kramabench.discovery")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ranker", choices=["rag", "bm25"], required=True)
    ap.add_argument("--mode", choices=["dense", "hybrid"], help="override retrieval.mode (rag)")
    ap.add_argument("--scope", choices=["pooled", "domain"], default="pooled")
    ap.add_argument("--out", type=Path, required=True)
    return asyncio.run(_main(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
