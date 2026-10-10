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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ds_research_agent.ledger import Batch, Ledger, Step
from eval.kramabench import agent_runs, failures
from eval.kramabench.baselines import PROGRAM_CONDITIONS
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


def compare(
    ledger: Ledger,
    base: Batch,
    other: Batch,
    tasks: dict[str, Task],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired difference ``other - base`` over their shared tasks."""
    if set(base.task_keys) != set(other.task_keys):
        raise ValueError("batches cover different tasks; a paired comparison needs the same set")
    ob, oo = outcomes(ledger, base, tasks), outcomes(ledger, other, tasks)
    out: dict[str, Any] = {"base": base.batch_id, "other": other.batch_id}
    for field in ("score", "strict"):
        b, o = per_task(ob, field), per_task(oo, field)
        ci = bootstrap([b, o], ob, resamples, seed)[2]
        out[field] = {
            "base": round(statistics.fmean(b.values()), 4),
            "other": round(statistics.fmean(o.values()), 4),
            "diff": round(statistics.fmean(o[k] - b[k] for k in b), 4),
            "diff_ci95": [round(ci[0], 4), round(ci[1], 4)],
        }
    return out


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
