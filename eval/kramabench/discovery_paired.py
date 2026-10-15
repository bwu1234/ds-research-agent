"""Paired differences between discovery results on the same tasks (no model).

Each pair is two ``eval.kramabench.discovery`` outputs, ``a`` and ``b``; the
difference ``a - b`` is taken per task, then averaged with a percentile
bootstrap of tasks within domain. Only tasks scored in both are paired.

    python -m eval.kramabench.discovery_paired \
        --pair data/measurements/discovery/v3-rag-hybrid-domain.json \
               data/measurements/discovery/rag-hybrid-v2-domain.json \
        --out data/measurements/discovery/paired-v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.kramabench.discovery import bootstrap_mean

METRICS = ("recall@5", "recall@10", "hit@5", "complete@10")
RESAMPLES = 2000


def paired(
    a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, Any]:
    rows_a = {r["task"]: r for r in a if r["gold"]}
    rows_b = {r["task"]: r for r in b if r["gold"]}
    common = sorted(rows_a.keys() & rows_b.keys())
    strata = {t: rows_a[t]["domain"] for t in common}
    out: dict[str, Any] = {"tasks": len(common)}
    for m in metrics:
        if not all(m in rows_a[t] and m in rows_b[t] for t in common):
            continue
        diffs = {t: float(rows_a[t][m]) - float(rows_b[t][m]) for t in common}
        mean, lo, hi = bootstrap_mean(diffs, strata, resamples=RESAMPLES)
        out[m] = [round(mean, 3), round(lo, 3), round(hi, 3)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kramabench.discovery_paired")
    ap.add_argument("--pair", nargs=2, type=Path, action="append", required=True)
    ap.add_argument("--metric", action="append", help=f"default: {', '.join(METRICS)}")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    metrics = args.metric or list(METRICS)
    pairs = []
    for pa, pb in args.pair:
        a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (pa, pb))
        pairs.append({"a": pa.stem, "b": pb.stem, **paired(a["tasks"], b["tasks"], metrics)})
        print(pairs[-1])
    result = {
        "uncertainty": f"paired a-b, percentile bootstrap of tasks within domain, "
        f"{RESAMPLES} resamples",
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
