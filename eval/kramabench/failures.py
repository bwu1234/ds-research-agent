"""The D3 failure taxonomy and a per-run rendering for reading failures.

Every failed run (not strictly correct) gets exactly one category from the
fixed taxonomy in ``docs/data-and-provenance.md``. Two categories follow from
the stop reason alone and are labelled by rule; the rest need a person (or
an agent acting as evaluator) to read the run and label it by hand. A rule
label never overwrites a manual one. Runs that are strictly correct but not
verified are verification failures, counted in the report's agent block,
not here.

``render_run`` is evaluator-side: it shows the gold answer, so its output
stays local like the ledger.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Literal, get_args

from ds_research_agent.ledger import FailureLabel, Ledger, Run, utc_now
from eval.kramabench.scoring import PROFILE
from eval.kramabench.tasks import Task

TAXONOMY = "taxonomy-v1"
Category = Literal[
    "discovery_miss",
    "parse_error",
    "wrong_filter_or_join",
    "formatting",
    "budget_exhausted",
    "tool_call_failure",
    "other",
]
CATEGORIES: tuple[str, ...] = get_args(Category)

_RULES = {
    "max_steps": "budget_exhausted",
    "max_wall": "budget_exhausted",
    "tool_call_failure": "tool_call_failure",
}


def failed(ledger: Ledger, run: Run) -> bool:
    sc = ledger.score(run.run_id, PROFILE)
    return sc is None or not sc.strict


def apply_rules(ledger: Ledger, batch_id: str) -> int:
    """Label failed runs whose stop reason decides the category; return how many."""
    n = 0
    for r in ledger.runs(batch_id):
        cat = _RULES.get(r.stop_reason or "")
        if cat is None or not failed(ledger, r):
            continue
        old = ledger.failure_label(r.run_id)
        if old is not None and old.source == "manual":
            continue
        ledger.put_failure_label(
            FailureLabel(
                run_id=r.run_id,
                taxonomy=TAXONOMY,
                category=cat,
                source="rule",
                note=f"stop reason {r.stop_reason}",
                labelled=utc_now(),
            )
        )
        n += 1
    return n


def label(ledger: Ledger, run: Run, category: str, note: str) -> FailureLabel:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; one of {', '.join(CATEGORIES)}")
    if not failed(ledger, run):
        raise ValueError(f"{run.run_id} is strictly correct; only failed runs are labelled")
    lab = FailureLabel(
        run_id=run.run_id,
        taxonomy=TAXONOMY,
        category=category,
        source="manual",
        note=note,
        labelled=utc_now(),
    )
    ledger.put_failure_label(lab)
    return lab


def counts(ledger: Ledger, runs: list[Run], tasks: dict[str, Task]) -> dict[str, Any]:
    """Category counts among failed runs, overall and by domain."""
    total: Counter[str] = Counter()
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for r in runs:
        if not failed(ledger, r):
            continue
        lab = ledger.failure_label(r.run_id)
        cat = lab.category if lab else "unlabelled"
        total[cat] += 1
        by_domain[tasks[r.task_key].domain][cat] += 1
    return {
        "taxonomy": TAXONOMY,
        "failed_runs": sum(total.values()),
        "unlabelled": total["unlabelled"],
        "counts": {c: total[c] for c in CATEGORIES if total[c]},
        "by_domain": {d: dict(sorted(c.items())) for d, c in sorted(by_domain.items())},
    }


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} chars]"


def render_run(ledger: Ledger, run: Run, task: Task, limit: int = 3000) -> str:
    """The run as the model saw it, then cell records, rerun, and verdicts."""
    sc = ledger.score(run.run_id, PROFILE)
    out = [
        f"# {run.run_id}",
        f"task {task.key} ({task.domain}, {task.difficulty}, answer type {task.answer_type})",
        f"stop {run.stop_reason}  wall {run.wall_s}s  error {run.error}",
        f"score {sc.score if sc else None}  strict {sc.strict if sc else None}  "
        f"verified {sc.verified_success if sc else None}",
        f"gold (evaluator only): {json.dumps(task.gold.answer)}",
        "resolution: "
        + "; ".join(
            f"{e['entry']} -> {e['tier']}" for e in run.input_manifest.get("resolution", [])
        ),
    ]
    steps = ledger.steps(run.run_id)
    if steps:
        last = steps[-1]
        msgs = list(last.request["messages"])
        if last.response:
            msgs.append(last.response["message"])
        out.append(f"\n## transcript ({len(steps)} model requests)")
        for i, m in enumerate(msgs):
            if m["role"] == "system":
                continue
            head = f"\n--- [{i}] {m['role']}" + (
                f" ({m['tool_name']})" if m.get("tool_name") else ""
            )
            out.append(head)
            if m.get("thinking"):
                out.append("(thinking) " + _clip(m["thinking"], limit))
            if m.get("content"):
                out.append(_clip(m["content"], limit))
            for c in m.get("tool_calls") or ():
                # Requests use Ollama's shape; the recorded response, ToolCall's.
                fn = c.get("function", c)
                args = dict(fn["arguments"])
                code = args.pop("code", None) or args.pop("program", None)
                out.append(f"> {fn['name']} {json.dumps(args, ensure_ascii=False)}")
                if code:
                    out.append(_clip(str(code), limit))
        out.append("\n## steps")
        for s in steps:
            out.append(
                f"{s.seq}: prompt {s.prompt_eval_tokens} tok {s.prompt_eval_s}s, "
                f"out {s.output_tokens} tok, wall {s.wall_s}s, {s.done_reason}"
                + (f", error {s.error}" if s.error else "")
            )
    progs = ledger.programs(run.run_id)
    if progs:
        out.append("\n## programs")
        for p in progs:
            out.append(
                f"{p.seq}: {p.kind} step {p.step} status {p.status} wall {p.wall_s}s "
                f"audit {'complete' if p.audit_complete else 'INCOMPLETE'} "
                f"reads {p.observed_reads}" + (" replan" if p.replan_requested else "")
            )
    sub = ledger.submission(run.run_id)
    if sub:
        out.append(f"\n## submission (step {sub.step})")
        out.append(f"answer {json.dumps(sub.value)}  files {sub.files_used}")
        out.extend(f"assumption: {a}" for a in sub.assumptions)
    v = ledger.verification(run.run_id)
    if v:
        out.append(
            f"\n## verification ({v.comparator})\nrerun exit {v.rerun_exit_code} "
            f"answer {json.dumps(v.rerun_answer)} reproduced {v.reproduced} "
            f"access {v.access_verified} detail {v.detail}"
        )
    lab = ledger.failure_label(run.run_id)
    if lab:
        out.append(f"\n## failure label: {lab.category} ({lab.source}) {lab.note}")
    return "\n".join(out)
