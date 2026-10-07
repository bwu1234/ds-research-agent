"""Development/holdout split of the KramaBench tasks, frozen by hash.

    uv run python -m eval.kramabench.split --config config/local.yaml show
    uv run python -m eval.kramabench.split --config config/local.yaml freeze
    uv run python -m eval.kramabench.split --config config/local.yaml verify

The split is a pure function of the pinned workload files, ``split_seed``,
``dev_fraction`` and ``sample_per_domain``, so it is never committed (task
lists are benchmark-derived). ``freeze`` writes it next to ``fetch.json`` and
prints its hash; put that hash in ``eval.split_sha256``. Every evaluation run
recomputes the split and refuses to start if the hash differs.

Rule, per domain: the quota is ``round_half_up(dev_fraction * n)``. It is
shared between easy and hard tasks by largest remainder (ties to the
difficulty with more tasks, then easy). Parents of smoke-test tasks are
placed in development first; the rest of each difficulty's quota is drawn
from a seeded shuffle. Sub-tasks stay with their parent because only parent
tasks are split. The D3 sample takes ``sample_per_domain`` development
tasks per domain whose sources all resolve, alternating easy and hard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ds_research_agent.config import Settings, load_settings
from eval.kramabench.tasks import (
    DOMAINS,
    Task,
    load_tasks,
    main_tasks,
    resolve_sources,
    workload_identity,
)

SPLIT_VERSION = 1


class Split(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    kramabench_commit: str
    workloads: dict[str, str]
    seed: int
    dev_fraction: float
    sample_per_domain: int
    dev: tuple[str, ...]
    holdout: tuple[str, ...]
    smoke: tuple[str, ...]
    sample: tuple[str, ...]

    def canonical(self) -> bytes:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=1).encode() + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()

    def split_of(self, key: str) -> str:
        if key in self.dev:
            return "dev"
        if key in self.holdout:
            return "holdout"
        if key in self.smoke:
            return "smoke"
        raise KeyError(key)


class SplitError(RuntimeError):
    pass


def _order(t: Task) -> int:
    return int(t.task_id.rsplit("-", 1)[1])


def _quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Share ``total`` across difficulties by largest remainder."""
    n = sum(counts.values())
    exact = {d: total * c / n for d, c in counts.items()}
    q = {d: math.floor(v) for d, v in exact.items()}
    rest = sorted(counts, key=lambda d: (-(exact[d] - q[d]), -counts[d], d))
    for d in rest[: total - sum(q.values())]:
        q[d] += 1
    return q


def make_split(
    tasks: dict[str, Task],
    *,
    seed: int,
    dev_fraction: float,
    sample_per_domain: int,
    kramabench_commit: str,
    workloads: dict[str, str],
    visible_root: Path,
) -> Split:
    rng = random.Random(seed)
    forced = {t.parent_key for t in tasks.values() if t.workload not in DOMAINS}
    dev: list[str] = []
    holdout: list[str] = []
    sample: list[str] = []
    for domain in DOMAINS:
        by_diff: dict[str, list[Task]] = defaultdict(list)
        for t in sorted(main_tasks(tasks), key=_order):
            if t.domain == domain:
                by_diff[t.difficulty].append(t)
        n = sum(len(v) for v in by_diff.values())
        quota = _quotas({d: len(v) for d, v in by_diff.items()}, math.floor(dev_fraction * n + 0.5))
        dom_dev: dict[str, list[Task]] = {}
        for diff in sorted(by_diff):
            pool = by_diff[diff]
            first = [t for t in pool if t.key in forced]
            others = [t for t in pool if t.key not in forced]
            rng.shuffle(others)
            if len(first) > quota[diff]:
                raise SplitError(f"{domain}/{diff}: more smoke parents than dev quota")
            chosen = (first + others)[: quota[diff]]
            dom_dev[diff] = chosen
            dev += [t.key for t in chosen]
            holdout += [t.key for t in pool if t not in chosen]
        # D3 sample: development tasks whose every source resolves.
        eligible = {
            diff: [
                t
                for t in sorted(ts, key=_order)
                if all(
                    r.tier != "unresolved"
                    for r in resolve_sources(t.domain, t.data_sources, visible_root)
                )
            ]
            for diff, ts in dom_dev.items()
        }
        for v in eligible.values():
            rng.shuffle(v)
        picked: list[str] = []
        order = [d for d in ("easy", "hard") if d in eligible]
        while len(picked) < sample_per_domain and any(eligible[d] for d in order):
            for d in order:
                if eligible[d] and len(picked) < sample_per_domain:
                    picked.append(eligible[d].pop(0).key)
        sample += picked
    key_order = {k: i for i, k in enumerate(tasks)}
    return Split(
        version=SPLIT_VERSION,
        kramabench_commit=kramabench_commit,
        workloads=workloads,
        seed=seed,
        dev_fraction=dev_fraction,
        sample_per_domain=sample_per_domain,
        dev=tuple(sorted(dev, key=key_order.__getitem__)),
        holdout=tuple(sorted(holdout, key=key_order.__getitem__)),
        smoke=tuple(t.key for t in tasks.values() if t.workload not in DOMAINS),
        sample=tuple(sorted(sample, key=key_order.__getitem__)),
    )


def split_for(settings: Settings) -> tuple[Split, dict[str, Task]]:
    kb, ev = settings.kramabench, settings.eval
    tasks = load_tasks(kb.evaluator_root)
    split = make_split(
        tasks,
        seed=ev.split_seed,
        dev_fraction=ev.dev_fraction,
        sample_per_domain=ev.sample_per_domain,
        kramabench_commit=kb.commit,
        workloads=workload_identity(kb.evaluator_root),
        visible_root=kb.visible_root,
    )
    return split, tasks


def frozen_split(settings: Settings) -> tuple[Split, dict[str, Task]]:
    """The split, refusing to proceed unless it matches ``eval.split_sha256``."""
    split, tasks = split_for(settings)
    want = settings.eval.split_sha256
    if want is None:
        raise SplitError("eval.split_sha256 is not set; run `split freeze` first")
    if split.sha256() != want:
        raise SplitError(f"split hash {split.sha256()} differs from frozen {want}")
    return split, tasks


def split_path(settings: Settings) -> Path:
    return settings.kramabench.evaluator_root.parent / "split.json"


def summary(split: Split, tasks: dict[str, Task]) -> dict[str, Any]:
    def count(keys: tuple[str, ...]) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for k in keys:
            c[tasks[k].domain] += 1
        return dict(sorted(c.items()))

    return {
        "sha256": split.sha256(),
        "dev": len(split.dev),
        "holdout": len(split.holdout),
        "smoke": len(split.smoke),
        "sample": len(split.sample),
        "dev_by_domain": count(split.dev),
        "holdout_by_domain": count(split.holdout),
        "dev_hard": sum(tasks[k].difficulty == "hard" for k in split.dev),
        "holdout_hard": sum(tasks[k].difficulty == "hard" for k in split.holdout),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kramabench.split")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("command", choices=["show", "freeze", "verify"])
    args = ap.parse_args(argv)
    settings = load_settings(args.config)
    split, tasks = split_for(settings)
    # Counts and hashes only; task lists stay in the ignored split file.
    print(json.dumps(summary(split, tasks), indent=2))
    if args.command == "freeze":
        want = settings.eval.split_sha256
        if want is not None and want != split.sha256():
            print(f"error: config freezes a different split ({want})", file=sys.stderr)
            return 1
        split_path(settings).write_bytes(split.canonical())
        print(f"wrote {split_path(settings)}; set eval.split_sha256: {split.sha256()}")
    elif args.command == "verify":
        try:
            frozen_split(settings)
        except SplitError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print("split matches eval.split_sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
