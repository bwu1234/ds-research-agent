"""Group-card granularity measurement (D2): no model, no index.

Candidate grouping rules partition each domain's catalogue entries into
groups (two or more members) and ungrouped files. Each rule is scored against
the development tasks' gold sets as an oracle: how many selections a task
needs, and how many extra files they bring. This bounds what any ranker over
that rule's cards could reach; it says nothing about whether a ranker would
find them.

Rules:

- ``directory``: the file's parent directory.
- ``schema``: format plus the column names of every table the profiler finds
  (files with no table are not grouped).
- ``pattern``: the file name with digit runs masked (``water-body-testing-#.csv``),
  across directories in a domain. Names without a digit run are not grouped
  (``Alabama.csv`` in two directories is not a family), and members must
  share the schema when the profiler finds one (numbered supplementary
  tables with different columns are not one table split up).
- ``directory+schema``: both keys.

- ``union``: every group the three single-key rules make; groups overlap.
- ``schema|pattern``: schema and pattern groups, without directory groups.
  In a flat directory of unrelated files, a directory group's contiguous
  runs are alphabetical accidents, not filters anyone would write.

Per task, three selection policies. A file may be selected on its own under
every policy (a file card is always available); what differs is which group
selections count:

- ``exact``: whole groups whose members are all gold. No extra files;
  ``exact_units`` selections.
- ``range``: a group restricted to one contiguous run of its members in name
  order, all gold (a date-range or glob filter on a group ID). No extra
  files; ``range_units`` selections.
- ``cover``: grouped gold files are selected only through a whole group (the
  one with the fewest extra files), ungrouped ones on their own.
  ``cover_units`` selections bring ``excess`` non-gold files.

Exact and range counts are a greedy set cover (optimal when groups do not
overlap; an upper bound for the overlapping rules).

``data_sources`` and the output are evaluator-side material: keep them local.

    python -m eval.kramabench.grouping --config config/local.kramabench-v2.yaml \
        --out data/measurements/discovery/grouping-v2.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ds_research_agent.catalogue.models import CatalogueEntry, load_manifest
from ds_research_agent.catalogue.profiler import PROFILERS, ProfileError
from ds_research_agent.config import CatalogueSettings, load_settings
from eval.kramabench.discovery import bootstrap_mean, gold_files
from eval.kramabench.split import frozen_split

# Column names per table; None when the profiler finds no table.
Schema = tuple[tuple[str, ...], ...] | None
Key = tuple[str, ...]
Rule = Callable[[CatalogueEntry, Schema], Key | None]

_DIGITS = re.compile(r"\d+")
LIMITS = (5, 10, 20)


def file_schema(entry: CatalogueEntry, root: Path, s: CatalogueSettings) -> Schema:
    profiler = PROFILERS.get(entry.format)
    if profiler is None or entry.parse_status != "ok":
        return None
    try:
        profile = profiler(root / entry.file_path, s)
    except ProfileError:
        return None
    tables = tuple(tuple(c.name for c in t.columns) for t in profile.tables if t.columns)
    return tables or None


def _directory(e: CatalogueEntry, _: Schema) -> Key:
    return (str(PurePosixPath(e.file_path).parent),)


def _schema(e: CatalogueEntry, schema: Schema) -> Key | None:
    if schema is None:
        return None
    return (e.domain, e.format, json.dumps(schema))


def _pattern(e: CatalogueEntry, schema: Schema) -> Key | None:
    name = PurePosixPath(e.file_path).name
    masked = _DIGITS.sub("#", name)
    if masked == name:
        return None
    return (e.domain, masked, json.dumps(schema))


def _directory_schema(e: CatalogueEntry, schema: Schema) -> Key | None:
    key = _schema(e, schema)
    return None if key is None else _directory(e, schema) + key


RULES: dict[str, Rule] = {
    "directory": _directory,
    "schema": _schema,
    "pattern": _pattern,
    "directory+schema": _directory_schema,
}
UNIONS = {
    "union": ("directory", "schema", "pattern"),
    "schema|pattern": ("schema", "pattern"),
}


def partition(
    entries: Sequence[CatalogueEntry], schemas: Mapping[str, Schema], rule: Rule
) -> list[tuple[str, ...]]:
    """Groups of two or more file paths, each sorted, in a stable order."""
    by: dict[Key, list[str]] = defaultdict(list)
    for e in entries:
        key = rule(e, schemas[e.file_path])
        if key is not None:
            by[key].append(e.file_path)
    return sorted(tuple(sorted(fs)) for fs in by.values() if len(fs) > 1)


def _runs(group: tuple[str, ...], want: set[str]) -> list[frozenset[str]]:
    """Maximal contiguous runs of gold members, in the group's (sorted) order."""
    runs: list[frozenset[str]] = []
    cur: list[str] = []
    for f in (*group, None):
        if f is not None and f in want:
            cur.append(f)
        elif cur:
            runs.append(frozenset(cur))
            cur = []
    return runs


def _set_cover(want: set[str], options: Sequence[frozenset[str]]) -> int:
    """Greedy count of options (each a subset of ``want``) plus single files."""
    left = set(want)
    units = 0
    for o in sorted(options, key=lambda o: (-len(o), sorted(o))):
        if len(o & left) > 1:
            left -= o
            units += 1
    return units + len(left)


def selection(
    groups: Sequence[tuple[str, ...]], gold: Sequence[str], sizes: Mapping[str, int]
) -> dict[str, float]:
    want = set(gold)
    touching = [g for g in groups if want.intersection(g)]
    exact_units = _set_cover(want, [frozenset(g) for g in touching if want.issuperset(g)])
    range_units = _set_cover(want, [r for g in touching for r in _runs(g, want)])

    containing: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for g in touching:
        for f in g:
            containing[f].append(g)
    chosen: set[tuple[str, ...]] = set()
    covered: set[str] = set()
    singles = 0
    for f in sorted(want):
        if f in covered:
            continue
        if not containing[f]:
            singles += 1
            covered.add(f)
            continue
        g = min(containing[f], key=lambda g: (len(set(g) - want), g))
        chosen.add(g)
        covered.update(g)
    cover_units = singles + len(chosen)
    excess = covered - want
    return {
        "gold": float(len(want)),
        "exact_units": float(exact_units),
        "range_units": float(range_units),
        "cover_units": float(cover_units),
        "excess": float(len(excess)),
        "excess_mb": round(sum(sizes[f] for f in excess) / 1e6, 3),
        **{
            f"{p}<={k}": float(u <= k)
            for p, u in (("exact", exact_units), ("range", range_units), ("cover", cover_units))
            for k in LIMITS
        },
        "zero_excess_cover": float(not excess),
    }


def group_stats(
    groups: Sequence[tuple[str, ...]], entries: Sequence[CatalogueEntry]
) -> dict[str, Any]:
    domain = {e.file_path: e.domain for e in entries}
    out: dict[str, Any] = {}
    for d in sorted({e.domain for e in entries}):
        gs = [g for g in groups if domain[g[0]] == d]
        sizes = sorted(len(g) for g in gs)
        files = sum(1 for e in entries if e.domain == d)
        out[d] = {
            "files": files,
            "groups": len(gs),
            "grouped_files": sum(sizes),
            "ungrouped_files": files - sum(sizes),
            "sizes": dict(sorted(Counter(sizes).items())),
        }
    return out


def measure(
    entries: Sequence[CatalogueEntry],
    schemas: Mapping[str, Schema],
    tasks: Sequence[tuple[str, str, tuple[str, ...]]],
) -> dict[str, Any]:
    """``tasks`` is (task key, domain, gold files)."""
    sizes = {e.file_path: e.size for e in entries}
    strata = {k: d for k, d, _ in tasks}
    baseline: list[tuple[str, ...]] = []  # per-file cards: no groups
    out: dict[str, Any] = {}
    partitions = {n: partition(entries, schemas, r) for n, r in RULES.items()}
    unions = {
        name: sorted({g for n in parts for g in partitions[n]}) for name, parts in UNIONS.items()
    }
    for name, groups in [("file", baseline), *partitions.items(), *unions.items()]:
        rows = {k: selection(groups, gold, sizes) for k, _, gold in tasks}
        metrics = next(iter(rows.values())).keys() if rows else ()
        summary = {}
        for m in metrics:
            mean, lo, hi = bootstrap_mean({k: r[m] for k, r in rows.items()}, strata)
            summary[m] = [round(mean, 3), round(lo, 3), round(hi, 3)]
        by_domain = {
            d: {m: round(sum(rows[k][m] for k, dk, _ in tasks if dk == d) / n, 3) for m in metrics}
            for d, n in sorted(Counter(strata.values()).items())
        }
        out[name] = {
            "groups": len(groups),
            "group_stats": group_stats(groups, entries),
            "summary": summary,
            "by_domain": by_domain,
            "tasks": rows,
        }
    return out


def _print(result: Mapping[str, Any]) -> None:
    cols = ["exact_units", "range_units", "cover_units", "excess", "range<=10", "exact<=10"]
    print(f"{'rule':18s}{'groups':>7s}" + "".join(f"{c:>19s}" for c in cols))
    for name, r in result["rules"].items():
        s = r["summary"]
        cells = "".join(
            f"{s[c][0]:>9.2f} [{s[c][1]:.2f},{s[c][2]:.2f}]"[:19].rjust(19) for c in cols
        )
        print(f"{name:18s}{r['groups']:>7d}{cells}")
    print("\nby domain (exact_units / range_units / cover_units / excess):")
    domains = next(iter(result["rules"].values()))["by_domain"]
    print(f"{'rule':18s}" + "".join(f"{d[:11]:>22s}" for d in domains))
    for name, r in result["rules"].items():
        cells = "".join(
            f"{v['exact_units']:.1f}/{v['range_units']:.1f}/{v['cover_units']:.1f}/{v['excess']:.0f}".rjust(
                22
            )
            for v in r["by_domain"].values()
        )
        print(f"{name:18s}{cells}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kramabench.grouping")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    settings = load_settings(args.config)
    manifest = load_manifest(settings.catalogue.manifest_path)
    root = settings.catalogue.data_root.resolve()

    start = time.monotonic()
    schemas = {e.file_path: file_schema(e, root, settings.catalogue) for e in manifest.entries}
    profile_s = time.monotonic() - start

    split, all_tasks = frozen_split(settings)
    visible = settings.kramabench.visible_root
    tasks = []
    for k in split.dev:
        gold, _ = gold_files(all_tasks[k], visible)
        if gold:
            tasks.append((k, all_tasks[k].domain, gold))

    result = {
        "catalogue": {
            "corpus": manifest.corpus,
            "profiler_version": manifest.profiler_version,
            "entries": len(manifest.entries),
            "without_schema": sum(1 for v in schemas.values() if v is None),
            "profile_s": round(profile_s, 1),
        },
        "tasks": len(tasks),
        "limits": list(LIMITS),
        "uncertainty": "mean [2.5%, 97.5%], percentile bootstrap of tasks within domain",
        "rules": measure(manifest.entries, schemas, tasks),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
