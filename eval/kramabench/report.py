"""Aggregate scores, uncertainty, operations, and a throughput model.

Every (task, repeat) the batch planned is reconciled against the ledger: a
missing run counts as zero with stop reason ``missing``, so the denominator is
always the planned task set. Repeats are averaged per task, tasks are equally
weighted, and uncertainty is a percentile bootstrap that resamples parent
tasks within each domain (sub-tasks, when added, travel with their parent).
Reports hold aggregates only, never answers or per-task results.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ds_research_agent.catalogue import Manifest, load_manifest
from ds_research_agent.ledger import Batch, Ledger, Step
from eval.kramabench import agent_runs, end_to_end, failures
from eval.kramabench.baselines import PROGRAM_CONDITIONS
from eval.kramabench.discovery import gold_files
from eval.kramabench.scoring import PROFILE
from eval.kramabench.tasks import Task

# Steps whose prompt evaluation took less than this are treated as cache hits
# and left out of the cold prefill rate.
COLD_PREFILL_MIN_S = 1.0


@dataclass(frozen=True)
class Outcome:
    key: str
    parent_key: str
    domain: str
    answer_type: str
    repeat: int
    score: float
    strict: bool
    # None in conditions without a program (the D1 baselines).
    verified: bool | None
    stop_reason: str
    parse_status: str | None
    wall_s: float | None
    prompt_tokens: int | None
    prompt_eval_s: float | None
    output_tokens: int | None
    eval_s: float | None
    thinking_chars: int | None


def _total(steps: Sequence[Step], field: str) -> Any:
    vals = [getattr(s, field) for s in steps if getattr(s, field) is not None]
    return sum(vals) if vals else None


def outcomes(ledger: Ledger, batch: Batch, tasks: dict[str, Task]) -> list[Outcome]:
    runs = {(r.task_key, r.repeat): r for r in ledger.runs(batch.batch_id)}
    out = []
    for key in batch.task_keys:
        t = tasks[key]
        for rep in range(batch.repeats):
            r = runs.get((key, rep))
            sc = ledger.score(r.run_id, PROFILE) if r else None
            ans = ledger.answer(r.run_id) if r else None
            steps = ledger.steps(r.run_id) if r else []

            out.append(
                Outcome(
                    key=key,
                    parent_key=t.parent_key,
                    domain=t.domain,
                    answer_type=t.answer_type,
                    repeat=rep,
                    score=sc.score if sc else 0.0,
                    strict=bool(sc and sc.strict),
                    verified=sc.verified_success if sc else None,
                    stop_reason=(r.stop_reason or "unfinished") if r else "missing",
                    parse_status=ans.parse_status if ans else None,
                    wall_s=r.wall_s if r else None,
                    prompt_tokens=_total(steps, "prompt_eval_tokens"),
                    prompt_eval_s=_total(steps, "prompt_eval_s"),
                    output_tokens=_total(steps, "output_tokens"),
                    eval_s=_total(steps, "eval_s"),
                    thinking_chars=_total(steps, "thinking_chars"),
                )
            )
    return out


def per_task(os_: Sequence[Outcome], field: str) -> dict[str, float]:
    """Mean over repeats of ``score`` or ``strict`` per task key."""
    acc: dict[str, list[float]] = defaultdict(list)
    for o in os_:
        acc[o.key].append(float(getattr(o, field)))
    return {k: statistics.fmean(v) for k, v in acc.items()}


def _strata(os_: Sequence[Outcome]) -> dict[str, list[str]]:
    """Domain -> parent keys -> (via key order) task keys, for resampling."""
    by_dom: dict[str, list[str]] = defaultdict(list)
    for o in os_:
        if o.parent_key not in by_dom[o.domain]:
            by_dom[o.domain].append(o.parent_key)
    return dict(sorted(by_dom.items()))


def bootstrap(
    values: Sequence[dict[str, float]],
    os_: Sequence[Outcome],
    resamples: int,
    seed: int,
) -> list[tuple[float, float]]:
    """95% percentile intervals for the mean of each of ``values`` (task
    key -> value), and of their paired differences if two are given, using
    shared resamples of parent tasks within domains."""
    parents: dict[str, list[str]] = defaultdict(list)
    for o in os_:
        if o.key not in parents[o.parent_key]:
            parents[o.parent_key].append(o.key)
    strata = _strata(os_)
    rng = random.Random(seed)
    series: list[list[float]] = [[] for _ in range(len(values) + (len(values) == 2))]
    for _ in range(resamples):
        keys = [k for ps in strata.values() for p in rng.choices(ps, k=len(ps)) for k in parents[p]]
        means = [statistics.fmean(v[k] for k in keys) for v in values]
        for i, m in enumerate(means):
            series[i].append(m)
        if len(values) == 2:
            series[2].append(means[1] - means[0])
    out = []
    for s in series:
        s.sort()
        out.append((s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]))
    return out


def _dist(xs: Sequence[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)
    return {
        "median": round(statistics.median(s), 2),
        "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 2),
        "mean": round(statistics.fmean(s), 2),
        "max": round(s[-1], 2),
        "total": round(sum(s), 1),
    }


def throughput(os_: Sequence[Outcome]) -> dict[str, Any]:
    cold = [o for o in os_ if o.prompt_eval_s and o.prompt_eval_s >= COLD_PREFILL_MIN_S]
    gen = [o for o in os_ if o.eval_s and o.output_tokens]
    prefill = (
        sum(o.prompt_tokens or 0 for o in cold) / sum(o.prompt_eval_s or 0 for o in cold)
        if cold
        else None
    )
    gen_rate = (
        sum(o.output_tokens or 0 for o in gen) / sum(o.eval_s or 0 for o in gen) if gen else None
    )
    walls = [o.wall_s for o in os_ if o.wall_s is not None]
    mean_wall = statistics.fmean(walls) if walls else None
    out_tokens = [o.output_tokens for o in gen if o.output_tokens]
    model: dict[str, Any] = {
        "prefill_tokens_per_s": round(prefill, 1) if prefill else None,
        "prefill_steps_used": len(cold),
        "generation_tokens_per_s": round(gen_rate, 1) if gen_rate else None,
        "projected_hours": {
            name: round(mean_wall * n / 3600, 2) if mean_wall else None
            for name, n in (("dev_53", 53), ("holdout_51", 51), ("all_104", 104))
        },
    }
    if prefill and gen_rate and out_tokens:
        # Agent step model for D3 budgets: each step prefills its new tokens
        # (previous output plus a tool result) and generates like this batch.
        per_step_out = statistics.median(out_tokens)
        tool_tokens = 1000
        step_s = (per_step_out + tool_tokens) / prefill + per_step_out / gen_rate
        model["agent_step_model"] = {
            "assumes": f"{tool_tokens} tool-output tokens per step, cached prefix, "
            f"median output {per_step_out:.0f} tokens per step from this batch",
            "seconds_per_step": round(step_s, 1),
            "minutes_per_task": {str(n): round(n * step_s / 60, 1) for n in (5, 10, 20)},
        }
    return model


def summarise(
    ledger: Ledger, batch: Batch, tasks: dict[str, Task], resamples: int, seed: int
) -> dict[str, Any]:
    os_ = outcomes(ledger, batch, tasks)
    scores, strict = per_task(os_, "score"), per_task(os_, "strict")
    (s_lo, s_hi), (a_lo, a_hi) = bootstrap([scores, strict], os_, resamples, seed)[:2]
    verified: Any = "not applicable (no program)"
    agent: dict[str, Any] | None = None
    fails: dict[str, Any] | None = None
    if batch.condition in PROGRAM_CONDITIONS:
        # A missing or unscored run counts as not verified.
        ver = per_task([replace(o, verified=bool(o.verified)) for o in os_], "verified")
        v_lo, v_hi = bootstrap([ver], os_, resamples, seed)[0]
        verified = {
            "rate": round(statistics.fmean(ver.values()), 4),
            "ci95": [round(v_lo, 4), round(v_hi, 4)],
            "definition": "strict and reproduced and observed access verified",
        }
        runs = ledger.runs(batch.batch_id)
        agent = agent_runs.summary_extra(ledger, [r.run_id for r in runs])
        fails = failures.counts(ledger, runs, tasks)
        rerun_s = agent["verifier_wall_s_per_run"]
        run_s = sum(r.wall_s for r in runs if r.wall_s is not None)
        agent["verifier_wall_s"] = _dist(rerun_s)
        agent["verifier_share_of_run_wall"] = round(sum(rerun_s) / run_s, 4) if run_s else None
        if batch.condition == end_to_end.CONDITION:
            agent["discovery"] = end_to_end.summary_extra(
                ledger, batch, tasks, *_catalogue_of(batch)
            )

    def group(attr: str) -> dict[str, Any]:
        keys: dict[str, set[str]] = defaultdict(set)
        for o in os_:
            keys[getattr(o, attr)].add(o.key)
        return {
            g: {
                "tasks": len(ks),
                "answer_score": round(statistics.fmean(scores[k] for k in ks), 4),
                "strict_accuracy": round(statistics.fmean(strict[k] for k in ks), 4),
            }
            for g, ks in sorted(keys.items())
        }

    return {
        "batch_id": batch.batch_id,
        "condition": batch.condition,
        "task_set": batch.task_set,
        "think": batch.model_settings.get("think"),
        "answer_type_visible": batch.answer_type_visible,
        "scoring_profile": PROFILE,
        "tasks": len(batch.task_keys),
        "repeats": batch.repeats,
        "runs_planned": len(os_),
        "runs_recorded": sum(o.stop_reason != "missing" for o in os_),
        "answer_score": round(statistics.fmean(scores.values()), 4),
        "answer_score_ci95": [round(s_lo, 4), round(s_hi, 4)],
        "strict_accuracy": round(statistics.fmean(strict.values()), 4),
        "strict_accuracy_ci95": [round(a_lo, 4), round(a_hi, 4)],
        "verified_success": verified,
        "uncertainty": f"percentile bootstrap, {resamples} resamples of parent tasks "
        f"within domain, seed {seed}",
        "by_domain": group("domain"),
        "by_answer_type": group("answer_type"),
        "stop_reasons": dict(Counter(o.stop_reason for o in os_)),
        "parse_status": dict(Counter(o.parse_status or "none" for o in os_)),
        "wall_s": _dist([o.wall_s for o in os_ if o.wall_s is not None]),
        "prompt_tokens": _dist([o.prompt_tokens for o in os_ if o.prompt_tokens is not None]),
        "output_tokens": _dist([o.output_tokens for o in os_ if o.output_tokens is not None]),
        "thinking_chars": _dist([o.thinking_chars for o in os_ if o.thinking_chars is not None]),
        "throughput": throughput(os_),
    } | ({"agent": agent, "failures": fails} if agent is not None else {})


def _catalogue_of(batch: Batch) -> tuple[Path, Manifest | None]:
    """Visible root and manifest as the batch recorded them; without the
    manifest, the seed search's reach is left out."""
    mpath = Path(batch.config["catalogue"]["manifest_path"])
    manifest = load_manifest(mpath) if mpath.exists() else None
    return Path(batch.config["kramabench"]["visible_root"]), manifest


# Selection states of an end-to-end run, from the furthest the files got.
SELECTION_STATES = ("selection_complete", "selection_incomplete", "seed_incomplete", "unscored")


def _selection_state(row: Mapping[str, Any]) -> str:
    if row.get("complete") == 1:
        return "selection_complete"
    if row.get("seed_complete") == 1:
        return "selection_incomplete"
    if row.get("seed_complete") == 0:
        return "seed_incomplete"
    return "unscored"


def run_selection(ledger: Ledger, run_id: str, tasks: dict[str, Task]) -> dict[str, Any]:
    """One end-to-end run's discovery, for reading a failure: its selection
    state and metrics, and the gold files missing from its selection and from
    its seed results (seed left out without the run's own catalogue)."""
    batch = ledger.batch(run_id.split("/", 1)[0])
    root, manifest = _catalogue_of(batch)
    row = end_to_end.selection_rows(ledger, batch, tasks, root, manifest)[run_id]
    run = next(r for r in ledger.runs(batch.batch_id) if r.run_id == run_id)
    gold, unresolved = gold_files(tasks[run.task_key], root)
    selected = set(end_to_end.selected_files(ledger, run_id))
    out = {"state": _selection_state(row), "gold_unresolved": unresolved} | row
    out["not_selected"] = [f for f in gold if f not in selected]
    out["extra_selected"] = sorted(selected - set(gold))
    if "seed_complete" in row and manifest is not None:
        k = int((batch.config.get("end_to_end") or {}).get("seed_cards", 0))
        collapse_groups = bool((batch.config.get("discovery") or {}).get("collapse_groups", True))
        reach = set(end_to_end.seed_files(ledger, run_id, manifest, k, collapse_groups))
        out["not_in_seed"] = [f for f in gold if f not in reach]
    return out


def by_selection(
    ledger: Ledger,
    base: Sequence[Batch],
    other: Sequence[Batch],
    tasks: dict[str, Task],
    resamples: int,
    seed: int,
    keys: Collection[str] | None = None,
) -> dict[str, Any]:
    """End-to-end strict accuracy split by how far each run's files got: the
    whole gold set selected, reachable from the seed search but not all
    selected, or not reachable. Each run is paired with ``base`` (given
    files, mean over its passes) on the same task, so a state's difference
    is net of how hard its tasks are. A task can fall in different states in
    different passes. Intervals resample parent tasks within domain, as
    ``bootstrap``; a state's runs come along with their parent."""
    keep = None if keys is None else set(keys)
    gf = per_task([o for b in base for o in outcomes(ledger, b, tasks)], "strict")
    items: dict[str, list[tuple[str, str, float, float]]] = defaultdict(list)
    for b in other:
        rows = end_to_end.selection_rows(ledger, b, tasks, *_catalogue_of(b))
        strict = {(o.key, o.repeat): o.strict for o in outcomes(ledger, b, tasks)}
        for r in ledger.runs(b.batch_id):
            if keep is not None and r.task_key not in keep:
                continue
            t = tasks[r.task_key]
            items[_selection_state(rows[r.run_id])].append(
                (t.domain, t.parent_key, float(strict[(r.task_key, r.repeat)]), gf[r.task_key])
            )
    rng = random.Random(seed)
    out: dict[str, Any] = {"base": [b.batch_id for b in base], "other": [b.batch_id for b in other]}
    for state in SELECTION_STATES:
        its = items.get(state, [])
        if not its:
            out[state] = {"runs": 0}
            continue
        by_parent: dict[str, list[tuple[float, float]]] = defaultdict(list)
        strata: dict[str, list[str]] = defaultdict(list)
        for dom, parent, e, g in its:
            by_parent[parent].append((e, g))
            if parent not in strata[dom]:
                strata[dom].append(parent)
        diffs = []
        for _ in range(resamples):
            drawn = [q for ps in strata.values() for q in rng.choices(ps, k=len(ps))]
            pairs = [p for q in drawn for p in by_parent[q]]
            diffs.append(statistics.fmean(e - g for e, g in pairs))
        diffs.sort()
        out[state] = {
            "runs": len(its),
            "parents": len(by_parent),
            "other_strict": round(statistics.fmean(e for _, _, e, _ in its), 4),
            "base_strict_same_tasks": round(statistics.fmean(g for *_, g in its), 4),
            "diff": round(statistics.fmean(e - g for _, _, e, g in its), 4),
            "diff_ci95": [
                round(diffs[int(0.025 * len(diffs))], 4),
                round(diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))], 4),
            ],
        }
    return out


def _median_task_wall(os_: Sequence[Outcome]) -> float | None:
    """Median over tasks of each task's mean wall time across its runs."""
    acc: dict[str, list[float]] = defaultdict(list)
    for o in os_:
        if o.wall_s is not None:
            acc[o.key].append(o.wall_s)
    return round(statistics.median(statistics.fmean(v) for v in acc.values()), 1) if acc else None


def compare(
    ledger: Ledger,
    base: Batch | Sequence[Batch],
    other: Batch | Sequence[Batch],
    tasks: dict[str, Task],
    resamples: int,
    seed: int,
    keys: Collection[str] | None = None,
) -> dict[str, Any]:
    """Paired difference ``other - base`` over their shared tasks, or over
    ``keys`` among them (D3: the untuned tasks, then the tuning sample).

    Either side may be several batches (passes of one condition): their runs
    are pooled, so each task's value is its mean over every pass (D3 (m),
    D4). Pooling narrows run-to-run noise, not task-sampling uncertainty."""
    bs = [base] if isinstance(base, Batch) else list(base)
    os_ = [other] if isinstance(other, Batch) else list(other)
    if not bs or not os_:
        raise ValueError("each side needs at least one batch")
    task_sets = {frozenset(b.task_keys) for b in bs + os_}
    if len(task_sets) != 1:
        raise ValueError("batches cover different tasks; a paired comparison needs the same set")
    (all_keys,) = task_sets
    keep = set(all_keys) if keys is None else set(keys)
    if not keep <= all_keys:
        raise ValueError("keys outside the batches' tasks")
    if not keep:
        raise ValueError("no tasks to compare")
    ob = [o for b in bs for o in outcomes(ledger, b, tasks) if o.key in keep]
    oo = [o for b in os_ for o in outcomes(ledger, b, tasks) if o.key in keep]
    out: dict[str, Any] = {
        "base": bs[0].batch_id if len(bs) == 1 else [b.batch_id for b in bs],
        "other": os_[0].batch_id if len(os_) == 1 else [b.batch_id for b in os_],
        "tasks": len(keep),
        "of_tasks": len(all_keys),
    }
    for field in ("score", "strict"):
        b, o = per_task(ob, field), per_task(oo, field)
        ci = bootstrap([b, o], ob, resamples, seed)[2]
        out[field] = {
            "base": round(statistics.fmean(b.values()), 4),
            "other": round(statistics.fmean(o.values()), 4),
            "diff": round(statistics.fmean(o[k] - b[k] for k in b), 4),
            "diff_ci95": [round(ci[0], 4), round(ci[1], 4)],
        }
    wb, wo = _median_task_wall(ob), _median_task_wall(oo)
    out["median_task_wall_s"] = {
        "base": wb,
        "other": wo,
        "ratio": round(wo / wb, 3) if wb and wo is not None else None,
    }
    return out


# Registered 2026-10-16 (docs/evaluation-plan.md, "End to end"), before any
# end-to-end model run: strict accuracy difference, 95% interval, against
# this margin, and the median task wall-time ratio against this limit.
D4_MARGIN = -0.10
D4_MAX_WALL_RATIO = 1.25


def d4_decision(cmp: Mapping[str, Any]) -> dict[str, Any]:
    """The registered D4 rule on a ``compare`` result (base: given files,
    other: end to end; untuned tasks; two passes a side). Retained if the
    interval's lower bound is above the margin and the wall ratio is within
    the limit; worse if its upper bound is below the margin; else
    inconclusive."""
    lo, hi = cmp["strict"]["diff_ci95"]
    ratio = cmp["median_task_wall_s"]["ratio"]
    if hi < D4_MARGIN:
        decision = "worse"
    elif lo > D4_MARGIN and ratio is not None and ratio <= D4_MAX_WALL_RATIO:
        decision = "retained"
    else:
        decision = "inconclusive"
    return {
        "rule": "D4 registered 2026-10-16: strict diff 95% interval vs margin, wall ratio limit",
        "margin": D4_MARGIN,
        "max_wall_ratio": D4_MAX_WALL_RATIO,
        "strict_diff_ci95": [lo, hi],
        "wall_ratio": ratio,
        "decision": decision,
    }


def choose_think(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered (2026-10-07, before the sweep ran) thinking-level rule.

    Among batches over the same task set, take the best mean answer score.
    Choose the level with the lowest median wall time whose score is no more
    than one task's worth (1 / tasks) below the best. Ties on time go to the
    higher score.
    """
    if len({s["task_set"] for s in summaries}) != 1:
        raise ValueError("the sweep batches must share a task set")
    best = max(s["answer_score"] for s in summaries)
    slack = 1 / summaries[0]["tasks"]
    ok = [s for s in summaries if s["answer_score"] >= best - slack]
    pick = min(ok, key=lambda s: (s["wall_s"]["median"], -s["answer_score"]))
    return {
        "rule": "fastest median wall time within 1/tasks of the best answer score",
        "best_answer_score": best,
        "slack": round(slack, 4),
        "eligible": [s["think"] for s in ok],
        "chosen": pick["think"],
    }
