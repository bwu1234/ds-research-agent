"""Viewer pages: batches, one batch, one run's trace, one task, and compare.

Numbers come from the report functions the CLI uses (``report.outcomes``,
``report.summarise``, ``report.compare``, ``discovery_agent.summary``), so a
page and ``run report`` cannot disagree. Pages quote prompts, model output,
data, and gold answers: like ``show``, they are local only.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

from ds_research_agent.config import Settings
from ds_research_agent.ledger import Batch, Ledger, LedgerError, Program, Run, Step
from eval.kramabench import discovery_agent, report
from eval.kramabench.baselines import PROGRAM_CONDITIONS
from eval.kramabench.discovery import gold_files
from eval.kramabench.scoring import PROFILE
from eval.kramabench.split import Split
from eval.kramabench.tasks import Task
from eval.kramabench.viewer import prompt, trace
from eval.kramabench.viewer.markup import (
    Child,
    Html,
    as_json,
    details,
    esc,
    join,
    kv,
    pre,
    table,
    tag,
    url,
)

DISCOVERY = discovery_agent.CONDITION


class NotFound(LookupError):
    pass


@dataclass
class Context:
    settings: Settings
    split: Split
    tasks: dict[str, Task]
    unseal_holdout: bool = False
    # (kind, batch IDs) -> (fingerprints, value); recomputed when a batch changes.
    cache: dict[tuple[str, ...], tuple[Any, Any]] = field(default_factory=dict)

    def cached[T](self, ledger: Ledger, kind: str, ids: Sequence[str], make: Callable[[], T]) -> T:
        key = (kind, *ids)
        fp = tuple(ledger.batch_fingerprint(b) for b in ids)
        hit = self.cache.get(key)
        if hit is not None and hit[0] == fp:
            cached: T = hit[1]
            return cached
        value = make()
        self.cache[key] = (fp, value)
        return value

    def sealed(self, task_set_or_split: str) -> bool:
        return task_set_or_split == "holdout" and not self.unseal_holdout

    def batch(self, ledger: Ledger, batch_id: str) -> Batch:
        try:
            b = ledger.batch(batch_id)
        except LedgerError as e:
            raise NotFound(str(e)) from None
        if self.sealed(b.task_set):
            raise NotFound(f"{batch_id} is a holdout batch; the holdout is sealed until D5")
        return b

    def task(self, key: str) -> Task:
        if key not in self.tasks:
            raise NotFound(f"no task {key!r}")
        if key in self.split.holdout and not self.unseal_holdout:
            raise NotFound(f"{key} is a holdout task; the holdout is sealed until D5")
        return self.tasks[key]


@dataclass(frozen=True)
class Page:
    title: str
    body: Html
    crumbs: tuple[tuple[str, str | None], ...] = ()


# ---- formatting -------------------------------------------------------------


def num(x: float | int | None, digits: int = 1) -> str:
    if x is None:
        return ""
    if isinstance(x, int) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def score(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def ci(pair: Sequence[float] | None) -> str:
    return "" if not pair else f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def badge(text: str, kind: str = "muted", title: str | None = None) -> Html:
    return tag("span", text, class_=f"badge {kind}", title=title)


def yes_no(v: bool | None, yes: str = "yes", no: str = "no") -> Html:
    if v is None:
        return badge("n/a")
    return badge(yes, "ok") if v else badge(no, "bad")


def stop_badge(reason: str | None) -> Html:
    r = reason or "unfinished"
    kind = {"submitted": "ok", "answered": "ok", "chosen": "ok", "missing": "bad"}.get(r, "warn")
    return badge(r, kind)


def think_label(think: object) -> str:
    """The model's think setting as batch names spell it (False is "off")."""
    if think is None:
        return ""
    if isinstance(think, bool):
        return "on" if think else "off"
    return str(think)


def when(ts: str | None) -> str:
    return "" if not ts else ts[:16].replace("T", " ")


def short(sha: str | None, n: int = 12) -> str:
    return "" if not sha else sha.removeprefix("sha256:")[:n]


def batch_link(batch_id: str) -> Html:
    return tag("a", batch_id, href=url("/batch", id=batch_id))


def run_link(run_id: str, text: Child = None) -> Html:
    return tag("a", text if text is not None else run_id, href=url("/run", id=run_id))


def task_link(key: str, text: Child = None) -> Html:
    return tag("a", text if text is not None else key, href=url("/task", key=key))


def section(title: Child, *body: Child, id_: str | None = None) -> Html:
    return tag("section", tag("h2", title), *body, id=id_)


def sortable(
    head: Iterable[Child], rows: Iterable[tuple[dict[str, Any], Iterable[Child]]], table_id: str
) -> Html:
    """A table the page script can sort (click a heading) and filter. Each row
    is (data attributes, cells); a cell's ``data-v`` (from ``cell``) sorts it."""
    trs = []
    for attrs, cells in rows:
        tds = [c if isinstance(c, Html) and c.startswith("<td") else tag("td", c) for c in cells]
        trs.append(tag("tr", tds, **{f"data_{k}": v for k, v in attrs.items()}))
    return tag(
        "table",
        tag("thead", tag("tr", [tag("th", h) for h in head])),
        tag("tbody", trs),
        id=table_id,
        class_="sortable",
    )


def cell(display: Child, sort: float | int | str | None) -> Html:
    return tag("td", display, data_v="" if sort is None else sort)


def filter_bar(table_id: str, outcomes: bool = True) -> Html:
    sel = None
    if outcomes:
        sel = tag(
            "select",
            [
                tag("option", t, value=v)
                for v, t in (
                    ("", "all runs"),
                    ("fail", "failed (not strict)"),
                    ("strict", "strictly correct"),
                    ("unverified", "strict, not verified"),
                )
            ],
            data_outcome_for=table_id,
        )
    return tag(
        "div",
        tag("input", type="search", placeholder="filter rows…", data_filter_for=table_id),
        sel,
        tag("span", "", class_="count", data_count_for=table_id),
        class_="filters",
    )


# ---- batches ----------------------------------------------------------------


def headline(ledger: Ledger, ctx: Context, b: Batch) -> dict[str, Any]:
    """The batch's mean scores, as ``report.summarise`` computes them (without
    the bootstrap), or the discovery summary's pooled means."""
    if b.condition == DISCOVERY:
        s = discovery_agent.summary(ledger, b, ctx.tasks, ctx.settings.kramabench.visible_root)
        pooled = s["summary"]["pooled"]
        return {
            "recorded": len(ledger.runs(b.batch_id)),
            "planned": len(b.task_keys) * b.repeats,
            "score": (pooled.get("recall") or [None])[0],
            "strict": (pooled.get("complete") or [None])[0],
            "verified": None,
            "metric": "file recall / complete",
        }
    os_ = report.outcomes(ledger, b, ctx.tasks)
    out: dict[str, Any] = {
        "recorded": sum(o.stop_reason != "missing" for o in os_),
        "planned": len(os_),
        "score": statistics.fmean(report.per_task(os_, "score").values()),
        "strict": statistics.fmean(report.per_task(os_, "strict").values()),
        "verified": None,
        "metric": "answer score / strict",
    }
    if b.condition in PROGRAM_CONDITIONS:
        ver = [replace(o, verified=bool(o.verified)) for o in os_]
        out["verified"] = statistics.fmean(report.per_task(ver, "verified").values())
    return out


def batches_page(ledger: Ledger, ctx: Context, q: Mapping[str, str]) -> Page:
    all_ = [b for b in ledger.batches() if not ctx.sealed(b.task_set)]
    replays: dict[str, list[str]] = {}
    for b in all_:
        if b.replay_of:
            replays.setdefault(b.replay_of, []).append(b.batch_id)
    cond, tset, show_replays = q.get("condition", ""), q.get("task_set", ""), q.get("replays")
    shown = [
        b
        for b in reversed(all_)
        if (not cond or b.condition == cond)
        and (not tset or b.task_set == tset)
        and (show_replays or not b.replay_of)
    ]

    def opts(name: str, values: Iterable[str], cur: str) -> Html:
        return tag(
            "select",
            tag("option", f"any {name.replace('_', ' ')}", value=""),
            [tag("option", v, value=v, selected=v == cur) for v in sorted(set(values))],
            name=name,
        )

    form = tag(
        "form",
        opts("condition", (b.condition for b in all_), cond),
        opts("task_set", (b.task_set for b in all_), tset),
        tag(
            "label",
            tag("input", type="checkbox", name="replays", value="1", checked=bool(show_replays)),
            " show replays",
        ),
        tag("button", "apply"),
        method="get",
        action="/",
        class_="filters",
    )
    rows = []
    for b in shown:
        h = ctx.cached(ledger, "headline", [b.batch_id], partial(headline, ledger, ctx, b))
        done = h["recorded"] == h["planned"]
        n_replays = len(replays.get(b.batch_id, []))
        rows.append(
            (
                {"condition": b.condition},
                [
                    cell(when(b.created), b.created),
                    batch_link(b.batch_id),
                    b.condition,
                    b.task_set,
                    think_label(b.model_settings.get("think")),
                    b.ollama_version or "",
                    cell(
                        join(
                            f"{h['recorded']}/{h['planned']}",
                            None if done else [" ", badge("incomplete", "warn")],
                        ),
                        h["recorded"],
                    ),
                    cell(score(h["score"]), h["score"]),
                    cell(score(h["strict"]), h["strict"]),
                    cell(score(h["verified"]), h["verified"]),
                    cell(str(n_replays or ""), n_replays),
                    tag("span", b.note or "", class_="note"),
                ],
            )
        )
    body = join(
        tag(
            "p",
            "Score is the mean answer score (file recall for discovery batches); strict is "
            "strict accuracy (complete selection for discovery). Missing and failed runs count "
            "as zero, as in ",
            tag("code", "run report"),
            ".",
            class_="hint",
        ),
        form,
        filter_bar("batches", outcomes=False),
        sortable(
            [
                "created",
                "batch",
                "condition",
                "tasks",
                "think",
                "Ollama",
                "runs",
                "score",
                "strict",
                "verified",
                "replays",
                "note",
            ],
            rows,
            "batches",
        ),
    )
    return Page("Batches", body)


def _batch_meta(ledger: Ledger, b: Batch) -> Html:
    replays = [x.batch_id for x in ledger.batches() if x.replay_of == b.batch_id]
    return kv(
        [
            ("condition", b.condition),
            ("task set", f"{b.task_set} ({len(b.task_keys)} tasks × {b.repeats})"),
            ("created", when(b.created)),
            ("model", b.model_name),
            ("model settings", tag("code", as_json(b.model_settings))),
            ("Ollama", b.ollama_version or "unknown"),
            (
                "repo revision",
                join(short(b.repo_revision), " ", badge("dirty", "warn") if b.repo_dirty else None),
            ),
            ("KramaBench commit", short(b.kramabench_commit)),
            ("split", short(b.split_sha256)),
            ("config", short(b.config_sha256)),
            ("system prompt", short(b.system_prompt_sha256)),
            ("scoring profile", b.scoring_profile),
            ("answer type visible", "yes" if b.answer_type_visible else "no"),
            ("replay of", batch_link(b.replay_of) if b.replay_of else "—"),
            ("replays", join([[batch_link(r), " "] for r in replays]) if replays else "—"),
            ("note", b.note or "—"),
        ]
    )


def _metric_table(rows: Iterable[tuple[str, Child, Child]]) -> Html:
    return table(["metric", "value", "95% interval"], rows, class_="metrics")


def _summary_block(s: dict[str, Any]) -> Html:
    ver = s["verified_success"]
    rows: list[tuple[str, Child, Child]] = [
        ("runs recorded", f"{s['runs_recorded']} of {s['runs_planned']}", ""),
        ("answer score", score(s["answer_score"]), ci(s["answer_score_ci95"])),
        ("strict accuracy", score(s["strict_accuracy"]), ci(s["strict_accuracy_ci95"])),
    ]
    if isinstance(ver, dict):
        rows.append(("verified success", score(ver["rate"]), ci(ver["ci95"])))
    agent = s.get("agent")
    if agent:
        n = s["runs_recorded"]
        rows += [
            ("submitted", f"{agent['submitted']} of {n}", ""),
            ("reproduced", f"{agent['reproduced']} of {n}", ""),
            ("access verified", f"{agent['access_verified']} of {n}", ""),
            ("re-plans requested", str(agent["replans_requested"]), ""),
            ("submissions returned for a fix", str(agent["submissions_returned_for_fix"]), ""),
        ]
    dom = table(
        ["domain", "tasks", "answer score", "strict"],
        [
            [d, v["tasks"], score(v["answer_score"]), score(v["strict_accuracy"])]
            for d, v in s["by_domain"].items()
        ],
    )
    counts = [
        tag("h3", "stop reasons"),
        tag("p", [[stop_badge(k), f" {v}  "] for k, v in sorted(s["stop_reasons"].items())]),
    ]
    fails = s.get("failures")
    if fails:
        counts += [
            tag("h3", f"failure labels ({fails['taxonomy']}, {fails['failed_runs']} failed runs)"),
            tag(
                "p",
                [[badge(k, "bad"), f" {v}  "] for k, v in fails["counts"].items()],
                [badge("unlabelled", "warn"), f" {fails['unlabelled']}"]
                if fails["unlabelled"]
                else None,
            ),
        ]
    walls = s.get("wall_s") or {}
    return join(
        tag(
            "div",
            tag("div", _metric_table(rows), tag("p", s["uncertainty"], class_="hint")),
            tag("div", tag("h3", "by domain"), dom),
            tag("div", *counts),
            class_="grid",
        ),
        tag(
            "p",
            f"wall s per run: median {num(walls.get('median'))}, p90 {num(walls.get('p90'))}, "
            f"total {num(walls.get('total'))}",
            class_="hint",
        ),
        details("full summary (as run report prints it)", pre(as_json(s))),
    )


def _discovery_block(s: dict[str, Any]) -> Html:
    pooled = s["pooled"]
    rows = [(k, score(v[0]), ci(v[1:])) for k, v in pooled.items() if isinstance(v, list)]
    dom_metrics = [m for m in ("recall", "complete", "precision") if m in pooled]
    dom = table(
        ["domain", "tasks", *dom_metrics],
        [
            [d, v["tasks"], *(score((v.get(m) or [None])[0]) for m in dom_metrics)]
            for d, v in s["by_domain"].items()
        ],
    )
    return join(
        tag(
            "div",
            tag("div", _metric_table(rows)),
            tag("div", tag("h3", "by domain"), dom),
            tag(
                "div",
                tag("h3", "stop reasons"),
                tag("p", [[stop_badge(k), f" {v}  "] for k, v in s["stop_reasons"].items()]),
            ),
            class_="grid",
        ),
        details("full summary (as run report prints it)", pre(as_json(s))),
    )


@dataclass(frozen=True)
class RunRow:
    run_id: str | None
    task: Task
    repeat: int
    score: float | None
    strict: bool
    verified: bool | None
    stop: str
    label: str | None
    wall_s: float | None
    requests: int
    cells: int
    prompt_tokens: int | None
    output_tokens: int | None


def run_rows(ledger: Ledger, ctx: Context, b: Batch) -> list[RunRow]:
    runs = {(r.task_key, r.repeat): r for r in ledger.runs(b.batch_id)}
    out = []
    for o in report.outcomes(ledger, b, ctx.tasks):
        r = runs.get((o.key, o.repeat))
        steps = ledger.steps(r.run_id) if r else []
        lab = ledger.failure_label(r.run_id) if r else None
        out.append(
            RunRow(
                run_id=r.run_id if r else None,
                task=ctx.tasks[o.key],
                repeat=o.repeat,
                score=o.score if r else None,
                strict=o.strict,
                verified=o.verified,
                stop=o.stop_reason,
                label=lab.category if lab else None,
                wall_s=o.wall_s,
                requests=sum(s.kind == "model" for s in steps),
                cells=sum(p.kind == "cell" for p in ledger.programs(r.run_id)) if r else 0,
                prompt_tokens=o.prompt_tokens,
                output_tokens=o.output_tokens,
            )
        )
    return out


def _outcome(strict: bool, verified: bool | None) -> str:
    if not strict:
        return "fail"
    return "unverified" if verified is False else "strict"


def batch_page(ledger: Ledger, ctx: Context, q: Mapping[str, str]) -> Page:
    b = ctx.batch(ledger, q.get("id", ""))
    ev = ctx.settings.eval
    if b.condition == DISCOVERY:
        root = ctx.settings.kramabench.visible_root
        full = ctx.cached(
            ledger,
            "discovery",
            [b.batch_id],
            lambda: discovery_agent.summary(ledger, b, ctx.tasks, root),
        )
        runs = {r.task_key: r for r in ledger.runs(b.batch_id) if r.repeat == 0}
        rows = []
        for row in full["tasks"]:
            r = runs.get(row["task"])
            t = ctx.tasks[row["task"]]
            rows.append(
                (
                    {"outcome": "strict" if row.get("complete") else "fail"},
                    [
                        run_link(r.run_id, t.task_id) if r else t.task_id,
                        t.domain,
                        cell(row["gold"], row["gold"]),
                        cell(score(row.get("recall")), row.get("recall")),
                        cell(score(row.get("complete")), row.get("complete")),
                        cell(score(row.get("precision")), row.get("precision")),
                        cell(num(row.get("searches"), 0), row.get("searches")),
                        cell(num(row.get("cards_read"), 0), row.get("cards_read")),
                        cell(num(row.get("model_requests"), 0), row.get("model_requests")),
                        stop_badge(r.stop_reason if r else "missing"),
                        cell(num(row.get("wall_s")), row.get("wall_s")),
                        task_link(t.key, "history"),
                    ],
                )
            )
        summary = _discovery_block(full["summary"])
        head = [
            "task",
            "domain",
            "gold files",
            "recall",
            "complete",
            "precision",
            "searches",
            "cards read",
            "requests",
            "stop",
            "wall s",
            "",
        ]
    else:
        s = ctx.cached(
            ledger,
            "summary",
            [b.batch_id],
            lambda: report.summarise(
                ledger, b, ctx.tasks, ev.bootstrap_resamples, ev.bootstrap_seed
            ),
        )
        rr = ctx.cached(ledger, "rows", [b.batch_id], lambda: run_rows(ledger, ctx, b))
        rows = [
            (
                {"outcome": _outcome(x.strict, x.verified)},
                [
                    run_link(x.run_id, x.task.task_id) if x.run_id else x.task.task_id,
                    x.task.domain,
                    x.task.difficulty,
                    x.task.answer_type,
                    cell(x.repeat, x.repeat),
                    cell(score(x.score), x.score),
                    cell(yes_no(x.strict), int(x.strict)),
                    cell(yes_no(x.verified), None if x.verified is None else int(x.verified)),
                    stop_badge(x.stop),
                    badge(x.label, "bad") if x.label else "",
                    cell(num(x.wall_s), x.wall_s),
                    cell(x.requests, x.requests),
                    cell(x.cells, x.cells),
                    cell(num(x.prompt_tokens), x.prompt_tokens),
                    cell(num(x.output_tokens), x.output_tokens),
                    task_link(x.task.key, "history"),
                ],
            )
            for x in rr
        ]
        summary = _summary_block(s)
        head = [
            "task",
            "domain",
            "difficulty",
            "answer type",
            "rep",
            "score",
            "strict",
            "verified",
            "stop",
            "failure label",
            "wall s",
            "requests",
            "cells",
            "prompt tok",
            "out tok",
            "",
        ]
    compare = (
        None
        if b.condition == DISCOVERY
        else tag("a", "compare with another batch →", href=url("/compare", base=b.batch_id))
    )
    body = join(
        section("Summary", summary, tag("p", compare)),
        section("Runs", filter_bar("runs"), sortable(head, rows, "runs")),
        section("Batch", _batch_meta(ledger, b)),
    )
    return Page(b.batch_id, body, (("Batches", "/"), (b.batch_id, None)))


# ---- one run ----------------------------------------------------------------


def _find_run(ledger: Ledger, ctx: Context, run_id: str) -> tuple[Batch, Run]:
    b = ctx.batch(ledger, run_id.split("/", 1)[0])
    for r in ledger.runs(b.batch_id):
        if r.run_id == run_id:
            if ctx.sealed(r.split):
                raise NotFound(f"{run_id} is a holdout run; the holdout is sealed until D5")
            return b, r
    raise NotFound(f"no run {run_id!r}")


def _text(content: str, class_: str = "text") -> Html | None:
    return pre(content, class_=class_) if content else None


def _program(p: Program, title: str | None = None) -> Html:
    rec = p.record
    status_ok = p.status in ("ok", "0", "complete")
    head = [
        tag("strong", title or p.kind.replace("_", " ")),
        " ",
        badge(p.status, "ok" if status_ok else "bad"),
        " ",
        badge(
            "audit complete" if p.audit_complete else "audit INCOMPLETE",
            "muted" if p.audit_complete else "bad",
        ),
        badge("re-plan requested", "warn") if p.replan_requested else None,
        tag("span", f" {num(p.wall_s, 2)} s", class_="hint"),
    ]
    mem = rec.get("memory_peak_bytes")
    if mem:
        head.append(tag("span", f" · peak {mem / 2**20:,.0f} MiB", class_="hint"))
    out = []
    for k in ("stdout", "stderr", "traceback"):
        v = rec.get(k)
        if v:
            trunc = " (truncated at capture)" if rec.get(f"{k}_truncated") else ""
            out.append(details(f"{k}{trunc} · {len(v):,} chars", pre(v, class_=k), open_=False))
    reads = p.observed_reads
    return tag(
        "div",
        tag("div", head, class_="prog-head"),
        tag("div", "reads: ", [[tag("code", r), " "] for r in reads], class_="reads")
        if reads
        else None,
        out,
        details("sandbox record", pre(as_json(rec))),
        class_="program",
    )


def _call(c: trace.CallView) -> Html:
    return tag(
        "div",
        tag(
            "div",
            "→ ",
            tag("strong", c.name),
            " ",
            tag("code", as_json(c.args).replace("\n", " ")) if c.args else None,
            class_="call-head",
        ),
        pre(c.code, class_="code") if c.code is not None else None,
        [_program(p, "ran as " + p.kind.replace("_", " ")) for p in c.programs],
        class_="call",
    )


def _message(m: trace.MessageView) -> Html:
    label = {"assistant": "model", "tool": f"tool result · {m.tool_name}", "user": "user"}.get(
        m.role, m.role
    )
    return tag(
        "div",
        tag("div", label, class_="role"),
        details(f"thinking · {len(m.thinking):,} chars", pre(m.thinking, class_="thinking"))
        if m.thinking
        else None,
        _text(m.content, "text" if m.role == "assistant" else "result"),
        [_call(c) for c in m.calls],
        class_=f"msg {m.role}",
    )


def _search(s: Step) -> Html:
    req, resp = s.request, s.response or {}
    results = resp.get("results") or []
    return details(
        join(
            "search ",
            tag("code", req.get("query", "")),
            f" · {len(results)} results · {num(s.wall_s, 2)} s",
            [" ", badge("error", "bad")] if s.error else None,
        ),
        kv(
            [
                ("filters", tag("code", as_json(req.get("filters")))),
                ("corpus", str(req.get("corpus"))),
                ("top k", str(req.get("top_k"))),
                ("candidates", str(resp.get("candidate_count", ""))),
                (
                    "raw",
                    tag(
                        "a", "request and response JSON", href=url("/step", id=s.run_id, seq=s.seq)
                    ),
                ),
                ("error", s.error or "—"),
            ]
        ),
        table(
            ["rank", "document", "score", "header", "text"],
            [
                [
                    r.get("rank"),
                    tag("code", r.get("document_id", "")),
                    score(r.get("score")),
                    r.get("header", ""),
                    tag("span", r.get("text", ""), class_="snippet"),
                ]
                for r in results
            ],
        ),
        class_="search",
    )


def _step(v: trace.StepView) -> Html:
    s = v.step
    status = []
    if not v.accepted:
        what = "error" if s.response is None else "rejected, repaired"
        status.append(badge(what, "bad" if s.response is None else "warn"))
    if s.done_reason and s.done_reason != "stop":
        status.append(badge(f"done: {s.done_reason}", "warn"))
    if s.tool_call_repairs:
        status.append(badge(f"{s.tool_call_repairs} repairs", "warn"))
    # Token counts are missing where the client does not report them (codeagent).
    parts = [f"t+{num(v.started_s)} s"]
    if s.prompt_eval_tokens is not None:
        parts.append(f"prompt {num(s.prompt_eval_tokens)} tok ({num(s.prompt_eval_s, 2)} s)")
    if s.output_tokens is not None:
        parts.append(f"out {num(s.output_tokens)} tok ({num(s.eval_s, 2)} s)")
    parts.append(f"wall {num(s.wall_s, 2)} s")
    metrics = " · ".join(parts)
    return tag(
        "article",
        tag(
            "header",
            tag("span", f"step {v.loop_step}", class_="step-n"),
            tag("span", f"request {v.ordinal}", class_="hint"),
            tag("a", "raw JSON", href=url("/step", id=s.run_id, seq=s.seq), class_="hint"),
            *status,
            tag("span", metrics, class_="metrics-line"),
        ),
        pre(s.error, class_="error") if s.error else None,
        [_message(m) for m in v.messages],
        [_program(p, "cell run from this reply") for p in v.programs],
        [_search(x) for x in v.searches],
        class_="step" + ("" if v.accepted else " rejected"),
        id=f"r{v.ordinal}",
    )


def _version(v: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in (v or "").split("-")[0].split(".")[:2])
    except ValueError:
        return ()


def _rendered(b: Batch, request: Mapping[str, Any]) -> tuple[prompt.Rendered | None, Html]:
    """The approximate prompt text and a note on how far to trust it."""
    try:
        r = prompt.render(b.model_name, request)
    except prompt.NoTemplate as e:
        return None, tag("p", badge("no template", "warn"), f" {e}.", class_="hint")
    except Exception as e:  # noqa: BLE001 - a template raise is shown, not fatal
        return None, tag("p", badge("render failed", "bad"), f" {e!r}", class_="hint")
    old = _version(b.ollama_version) < (0, 40)
    note = tag(
        "p",
        badge("approximate", "warn"),
        " Ollama builds this text inside /api/chat and does not return it. This is the "
        f"recorded request rendered through the model's own chat template ({r.template}) "
        f"with the tool and argument encoding of the {prompt.RENDERER} renderer. On Ollama "
        "0.40.4 batches its token count matched Ollama's recorded prompt count on all "
        "1,185 requests checked (2026-10-15). The ledger keeps JSON keys sorted, so tool "
        "properties and call arguments may appear in a different order than sent.",
        [
            " ",
            badge(f"Ollama {b.ollama_version}", "bad"),
            " This batch ran on an older renderer, which differed (exact on single requests, "
            "not on multi-turn agent requests).",
        ]
        if old
        else None,
        class_="hint",
    )
    return r, note


def _sent(tr: trace.Trace, b: Batch) -> Html:
    """What every request carries besides the conversation: the thinking
    setting, options, system prompt, and tool definitions."""
    first = tr.steps[0].step
    rendered, note = _rendered(b, first.request)
    off = tr.think is False
    specs = [t.get("function") or {} for t in tr.tool_specs]
    tools = [
        tag(
            "div",
            tag("div", tag("strong", fn.get("name", "")), class_="call-head"),
            pre(fn.get("description", ""), class_="text"),
            details("parameters (JSON Schema)", pre(as_json(fn.get("parameters")))),
            class_="call",
        )
        for fn in specs
    ]
    chars = len(as_json(tr.tool_specs)) if tr.tool_specs else 0
    return tag(
        "div",
        tag("h3", "Sent with every request"),
        kv(
            [
                (
                    "thinking",
                    join(
                        badge(think_label(tr.think) or "not set", "warn" if off else "muted"),
                        " Thinking was off for this batch, so replies carry no thinking."
                        if off
                        else None,
                    ),
                ),
                ("options", tag("code", as_json(tr.options).replace("\n", " "))),
                (
                    "raw request",
                    tag(
                        "a",
                        f"request 1 as recorded (JSON, {len(as_json(first.request)):,} chars)",
                        href=url("/step", id=first.run_id, seq=first.seq),
                    ),
                ),
            ]
        ),
        details(
            "system turn as the model read it · "
            + (f"{len(rendered.system_turn):,} chars" if rendered else "not available"),
            note,
            pre(rendered.system_turn, class_="text") if rendered else None,
            tag(
                "a",
                "full prompt text of request 1",
                href=url("/step", id=first.run_id, seq=first.seq) + "#prompt",
            )
            if rendered
            else None,
            open_=rendered is not None,
        ),
        details(
            f"system prompt as sent · {len(tr.system or ''):,} chars",
            pre(tr.system or "", class_="text"),
        ),
        details(
            f"tool definitions as sent · {', '.join(tr.tools)} · {chars:,} chars",
            tag(
                "p",
                "Sent in the request's tools field, beside the messages, not inside the "
                "system prompt; the renderer merges them into the system turn above.",
                class_="hint",
            ),
            tools,
        )
        if specs
        else tag("p", "No tools were offered (the model writes code in its reply).", class_="hint"),
        class_="sent",
    )


def _trace(tr: trace.Trace, b: Batch) -> Html:
    if not tr.steps:
        return tag("p", "No model requests were recorded for this run.", class_="hint")
    total = len(tr.steps)
    rejected = sum(not v.accepted for v in tr.steps)
    cells = sum(len(c.programs) for v in tr.steps for m in v.messages for c in m.calls) + sum(
        len(v.programs) for v in tr.steps
    )
    nav = tag(
        "nav",
        "jump to request: ",
        [[tag("a", str(v.ordinal), href=f"#r{v.ordinal}"), " "] for v in tr.steps],
        class_="jump",
    )
    return join(
        tag(
            "p",
            f"{total} model requests ({rejected} rejected or failed), {cells} programs. ",
            "Each step shows what it added to the context: the model's reply, then the "
            "results it got back, exactly as the model saw them. Program boxes add the "
            "sandbox's full record.",
            class_="hint",
        ),
        None
        if tr.append_only
        else tag(
            "p",
            badge("not append-only", "warn"),
            " This run's requests do not extend each other, so each request after the "
            "first is shown in full, followed by its reply.",
        ),
        nav if total > 3 else None,
        _sent(tr, b),
        [_message(m) for m in tr.preamble],
        [_step(v) for v in tr.steps],
        [
            tag(
                "div",
                badge("unplaced", "warn"),
                " These programs did not match a recorded tool call:",
                [_program(p) for p in tr.unplaced],
            )
        ]
        if tr.unplaced
        else None,
    )


def _evaluator(ledger: Ledger, ctx: Context, r: Run, t: Task) -> Html:
    if r.condition == DISCOVERY:
        gold, unresolved = gold_files(t, ctx.settings.kramabench.visible_root)
        a = ledger.answer(r.run_id)
        chosen = set(a.value.get("files", [])) if a and isinstance(a.value, dict) else set()
        rows = [[tag("code", f), yes_no(f in chosen, "chosen", "missed")] for f in sorted(gold)] + [
            [tag("code", f), badge("extra", "warn")] for f in sorted(chosen - set(gold))
        ]
        body = join(
            kv(
                [
                    ("data sources", tag("code", as_json(list(t.data_sources)))),
                    ("unresolved entries", str(unresolved)),
                ]
            ),
            table(["file", "selection"], rows),
        )
    else:
        body = kv([("gold answer", pre(as_json(t.gold.answer))), ("answer type", t.answer_type)])
    return details("Evaluator only: gold " + ("files" if r.condition == DISCOVERY else "answer"),
                   body, class_="evaluator")  # fmt: skip


def _inputs(r: Run) -> Html:
    m = r.input_manifest
    files = m.get("files")
    parts: list[Child] = []
    if isinstance(files, list) and files:
        parts.append(
            table(
                ["file", "bytes", "sha256"],
                [
                    [
                        tag("code", f.get("path", "")),
                        num(f.get("bytes")),
                        tag("code", short(f.get("sha256"))),
                    ]
                    for f in files
                ],
            )
        )
    res = m.get("resolution")
    if isinstance(res, list) and res:
        parts.append(
            tag(
                "p",
                "resolution: ",
                [[tag("code", e.get("entry")), f" → {e.get('tier')}  "] for e in res],
                class_="hint",
            )
        )
    parts.append(details("input manifest", pre(as_json(m))))
    return join(*parts)


def _submission(ledger: Ledger, r: Run, final: Sequence[Program]) -> Html | None:
    sub, v, a = ledger.submission(r.run_id), ledger.verification(r.run_id), ledger.answer(r.run_id)
    parts: list[Child] = []
    if a is not None:
        parts.append(
            kv(
                [
                    ("answer", pre(as_json(a.value))),
                    ("answered", yes_no(a.answered)),
                    ("parse status", a.parse_status),
                    ("abstained", "yes" if a.abstained else "no"),
                ]
            )
        )
    if sub is not None:
        parts += [
            tag("h3", f"submission (step {sub.step})"),
            kv(
                [
                    ("files used", [[tag("code", f), " "] for f in sub.files_used] or "—"),
                    (
                        "assumptions",
                        tag("ul", [tag("li", x) for x in sub.assumptions])
                        if sub.assumptions
                        else "—",
                    ),
                ]
            ),
            pre(sub.program, class_="code"),
        ]
    if v is not None:
        observed, claimed = set(v.observed_files), set(v.claimed_files)
        files = [
            [
                tag("code", f),
                yes_no(f in claimed, "claimed", "not claimed"),
                yes_no(f in observed, "read", "not read"),
            ]
            for f in sorted(observed | claimed)
        ]
        parts += [
            tag("h3", f"verification ({v.comparator})"),
            kv(
                [
                    ("rerun exit code", str(v.rerun_exit_code)),
                    ("rerun answered", yes_no(v.rerun_answered)),
                    ("rerun answer", pre(as_json(v.rerun_answer))),
                    ("reproduced", yes_no(v.reproduced)),
                    ("read audit complete", yes_no(v.audit_complete)),
                    ("access verified", yes_no(v.access_verified)),
                    ("detail", tag("ul", [tag("li", d) for d in v.detail]) if v.detail else "—"),
                ]
            ),
            table(["file", "claimed by the model", "observed by the runner"], files)
            if files
            else None,
        ]
    parts += [_program(p, "final rerun") for p in final]
    return join(*parts) if parts else None


def run_page(ledger: Ledger, ctx: Context, q: Mapping[str, str]) -> Page:
    b, r = _find_run(ledger, ctx, q.get("id", ""))
    t = ctx.tasks[r.task_key]
    profile = discovery_agent.PROFILE if r.condition == DISCOVERY else PROFILE
    sc = ledger.score(r.run_id, profile)
    lab = ledger.failure_label(r.run_id)
    tr = trace.build(ledger.steps(r.run_id), ledger.programs(r.run_id))
    facts = kv(
        [
            ("task", join(task_link(t.key), f" · {t.domain} · {t.difficulty} · {t.answer_type}")),
            ("condition", f"{r.condition} · repeat {r.repeat} · {r.access_policy}"),
            ("stop", join(stop_badge(r.stop_reason), f"  {num(r.wall_s)} s wall")),
            ("error", pre(r.error, class_="error") if r.error else "—"),
            (
                f"score ({profile})",
                join(
                    score(sc.score) if sc else "unscored",
                    f" ({sc.metric})" if sc else "",
                    "  strict ",
                    yes_no(sc.strict if sc else None),
                    "  verified ",
                    yes_no(sc.verified_success if sc else None),
                )
                if sc
                else badge("unscored", "warn"),
            ),
            (
                "failure label",
                join(badge(lab.category, "bad"), f" ({lab.source}) {lab.note}") if lab else "—",
            ),
            ("started", f"{when(r.started)} → {when(r.ended)}"),
            ("repo revision", short(r.repo_revision or b.repo_revision)),
            ("catalogue", r.catalogue_generation or "—"),
        ]
    )
    sub = _submission(ledger, r, tr.final)
    body = join(
        facts,
        tag("blockquote", t.query, class_="query"),
        _evaluator(ledger, ctx, r, t),
        section("Inputs", _inputs(r)),
        section("Trace", _trace(tr, b), id_="trace"),
        section("Answer and verification", sub) if sub else None,
        section("Sandbox session", [_program(p) for p in tr.session]) if tr.session else None,
    )
    crumbs = (("Batches", "/"), (b.batch_id, url("/batch", id=b.batch_id)), (t.task_id, None))
    return Page(f"{t.task_id} · {b.batch_id}", body, crumbs)


def _prompt_section(b: Batch, s: Step) -> Html | None:
    if s.kind != "model":
        return None
    rendered, note = _rendered(b, s.request)
    title = "Prompt text" + (f" · {len(rendered.text):,} chars" if rendered else "")
    if rendered is not None:
        v = rendered.variables
        note = join(
            note,
            tag(
                "p",
                f"Template variables from think={think_label(s.request.get('think')) or 'unset'}: ",
                tag("code", as_json(v).replace("\n", " ")),
                class_="hint",
            ),
        )
    body = pre(rendered.text, class_="raw") if rendered else None
    return section(title, note, body, id_="prompt")


def step_page(ledger: Ledger, ctx: Context, q: Mapping[str, str]) -> Page:
    """One recorded request and its response, exactly as stored and hashed."""
    b, r = _find_run(ledger, ctx, q.get("id", ""))
    try:
        seq = int(q.get("seq", ""))
    except ValueError:
        raise NotFound("seq must be a number") from None
    s = next((x for x in ledger.steps(r.run_id) if x.seq == seq), None)
    if s is None:
        raise NotFound(f"{r.run_id} has no step {seq}")
    t = ctx.tasks[r.task_key]
    request = as_json(s.request)
    body = join(
        kv(
            [
                ("run", run_link(r.run_id)),
                ("kind", s.kind),
                ("request sha256", tag("code", s.request_sha256)),
                ("done reason", s.done_reason or "—"),
                ("error", s.error or "—"),
            ]
        ),
        tag(
            "p",
            "The request as recorded and hashed (for model steps: messages, tools, think, "
            "options; context is append-only, so it holds the whole conversation so far), "
            "and the response as returned.",
            class_="hint",
        ),
        section(f"Request · {len(request):,} chars", pre(request, class_="raw")),
        section("Response", pre(as_json(s.response), class_="raw")),
        _prompt_section(b, s),
    )
    crumbs = (
        ("Batches", "/"),
        (b.batch_id, url("/batch", id=b.batch_id)),
        (t.task_id, url("/run", id=r.run_id)),
        (f"step {seq}", None),
    )
    return Page(f"{t.task_id} · step {seq} · raw", body, crumbs)


# ---- one task across batches ------------------------------------------------


def task_page(ledger: Ledger, ctx: Context, q: Mapping[str, str]) -> Page:
    t = ctx.task(q.get("key", ""))
    batches = {b.batch_id: b for b in ledger.batches()}
    rows = []
    for r in ledger.runs_for_task(t.key):
        b = batches[r.batch_id]
        if ctx.sealed(b.task_set) or ctx.sealed(r.split):
            continue
        profile = discovery_agent.PROFILE if r.condition == DISCOVERY else PROFILE
        sc = ledger.score(r.run_id, profile)
        rows.append(
            (
                {"outcome": _outcome(bool(sc and sc.strict), sc.verified_success if sc else None)},
                [
                    cell(when(r.started), r.started),
                    batch_link(b.batch_id),
                    r.condition,
                    b.task_set,
                    cell(r.repeat, r.repeat),
                    cell(score(sc.score) if sc else "", sc.score if sc else None),
                    cell(yes_no(sc.strict if sc else None), int(bool(sc and sc.strict))),
                    yes_no(sc.verified_success if sc else None),
                    stop_badge(r.stop_reason),
                    cell(num(r.wall_s), r.wall_s),
                    run_link(r.run_id, "trace"),
                ],
            )
        )
    splits = [
        n for n in ("dev", "sample", "smoke", "holdout") if t.key in getattr(ctx.split, n)
    ]  # fmt: skip
    body = join(
        kv(
            [
                ("domain", t.domain),
                ("difficulty", t.difficulty),
                ("answer type", t.answer_type),
                ("split", ", ".join(splits) or "—"),
                ("sub-tasks", str(t.subtask_count)),
            ]
        ),
        tag("blockquote", t.query, class_="query"),
        details(
            "Evaluator only: gold answer and data sources",
            kv(
                [
                    ("gold answer", pre(as_json(t.gold.answer))),
                    ("data sources", tag("code", as_json(list(t.data_sources)))),
                ]
            ),
            class_="evaluator",
        ),
        section(
            f"Runs ({len(rows)})",
            filter_bar("task-runs"),
            sortable(
                [
                    "started",
                    "batch",
                    "condition",
                    "tasks",
                    "rep",
                    "score",
                    "strict",
                    "verified",
                    "stop",
                    "wall s",
                    "",
                ],
                rows,
                "task-runs",
            ),
        ),
    )
    return Page(t.key, body, (("Batches", "/"), (t.key, None)))


# ---- compare ----------------------------------------------------------------


def compare_page(ledger: Ledger, ctx: Context, q: Mapping[str, str]) -> Page:
    choices = [
        b
        for b in reversed(ledger.batches())
        if b.condition != DISCOVERY and not ctx.sealed(b.task_set)
    ]
    base_id, other_id, subset = q.get("base", ""), q.get("other", ""), q.get("subset", "all")

    def select(name: str, cur: str) -> Html:
        return tag(
            "select",
            tag("option", f"{name}…", value=""),
            [
                tag(
                    "option",
                    f"{b.batch_id} ({b.condition}, {b.task_set})",
                    value=b.batch_id,
                    selected=b.batch_id == cur,
                )
                for b in choices
            ],
            name=name,
        )

    form = tag(
        "form",
        select("base", base_id),
        " vs ",
        select("other", other_id),
        tag(
            "select",
            [
                tag("option", label, value=v, selected=v == subset)
                for v, label in (
                    ("all", "all shared tasks"),
                    ("untuned", "untuned (not the tuning sample)"),
                    ("sample", "tuning sample only"),
                )
            ],
            name="subset",
        ),
        tag("button", "compare"),
        method="get",
        action="/compare",
        class_="filters",
    )
    parts: list[Child] = [
        tag(
            "p",
            "Paired difference other − base over the batches' shared tasks, from ",
            tag("code", "report.compare"),
            " (as ",
            tag("code", "run compare"),
            " prints it). Discovery batches are paired with ",
            tag("code", "discovery_paired"),
            " instead.",
            class_="hint",
        ),
        form,
    ]
    crumbs: tuple[tuple[str, str | None], ...] = (("Batches", "/"), ("Compare", None))
    if not (base_id and other_id):
        return Page("Compare", join(*parts), crumbs)
    base, other = ctx.batch(ledger, base_id), ctx.batch(ledger, other_id)
    if DISCOVERY in (base.condition, other.condition):
        raise NotFound("discovery batches are paired with eval.kramabench.discovery_paired")
    sample = set(ctx.split.sample)
    keys = {
        "all": None,
        "untuned": [k for k in base.task_keys if k not in sample],
        "sample": [k for k in base.task_keys if k in sample],
    }.get(subset)
    ev = ctx.settings.eval
    try:
        out = ctx.cached(
            ledger,
            f"compare-{subset}",
            [base.batch_id, other.batch_id],
            lambda: report.compare(
                ledger, base, other, ctx.tasks, ev.bootstrap_resamples, ev.bootstrap_seed, keys
            ),
        )
    except ValueError as e:
        parts.append(tag("p", badge("cannot compare", "bad"), f" {e}"))
        return Page("Compare", join(*parts), crumbs)
    if base.ollama_version is None or base.ollama_version != other.ollama_version:
        parts.append(
            tag(
                "p",
                badge("warning", "warn"),
                f" Ollama versions differ or are unknown: {base.ollama_version} (base) vs "
                f"{other.ollama_version} (other).",
            )
        )
    parts.append(
        table(
            ["metric", "base", "other", "other − base", "95% interval"],
            [
                [
                    name,
                    score(out[f]["base"]),
                    score(out[f]["other"]),
                    f"{out[f]['diff']:+.3f}",
                    ci(out[f]["diff_ci95"]),
                ]
                for f, name in (("score", "answer score"), ("strict", "strict accuracy"))
            ],
            class_="metrics",
        )
    )
    parts.append(
        tag(
            "p",
            f"{out['tasks']} of {out['of_tasks']} tasks; percentile bootstrap of parent tasks "
            "within domain. ",
            tag("a", "swap", href=url("/compare", base=other_id, other=base_id, subset=subset)),
            class_="hint",
        )
    )
    keep = set(base.task_keys) if keys is None else set(keys)
    ob = [o for o in report.outcomes(ledger, base, ctx.tasks) if o.key in keep]
    oo = [o for o in report.outcomes(ledger, other, ctx.tasks) if o.key in keep]
    sb, so = report.per_task(ob, "score"), report.per_task(oo, "score")
    rb = {r.task_key: r.run_id for r in ledger.runs(base.batch_id) if r.repeat == 0}
    ro = {r.task_key: r.run_id for r in ledger.runs(other.batch_id) if r.repeat == 0}
    rows = []
    for k in sorted(sb, key=lambda k: so[k] - sb[k]):
        t, d = ctx.tasks[k], so[k] - sb[k]
        kind = "fail" if d < 0 else "strict" if d > 0 else ""
        rows.append(
            (
                {"outcome": kind},
                [
                    task_link(k, t.task_id),
                    t.domain,
                    cell(run_link(rb[k], score(sb[k])) if k in rb else score(sb[k]), sb[k]),
                    cell(run_link(ro[k], score(so[k])) if k in ro else score(so[k]), so[k]),
                    cell(
                        tag("span", f"{d:+.3f}", class_="neg" if d < 0 else "pos" if d else None),
                        d,
                    ),
                ],
            )
        )
    parts.append(
        section(
            "Per task",
            tag(
                "p",
                "Mean over repeats; links open repeat 0. Sorted with the largest losses first.",
                class_="hint",
            ),
            filter_bar("pairs", outcomes=False),
            sortable(
                ["task", "domain", "base score", "other score", "other − base"], rows, "pairs"
            ),
        )
    )
    return Page("Compare", join(*parts), crumbs)


ROUTES: dict[str, Callable[[Ledger, Context, Mapping[str, str]], Page]] = {
    "/": batches_page,
    "/batch": batch_page,
    "/run": run_page,
    "/step": step_page,
    "/task": task_page,
    "/compare": compare_page,
}


def layout(page: Page) -> str:
    crumbs = join(
        [
            [tag("a", name, href=href) if href else tag("span", name), " › "]
            for name, href in page.crumbs[:-1]
        ],
        tag("span", page.crumbs[-1][0]) if page.crumbs else None,
    )
    return "<!doctype html>\n" + tag(
        "html",
        tag(
            "head",
            tag("meta", charset="utf-8"),
            tag("meta", name="viewport", content="width=device-width, initial-scale=1"),
            tag("title", f"{page.title} · run viewer"),
            tag("link", rel="stylesheet", href="/static/viewer.css"),
            tag("script", src="/static/viewer.js", defer=True),
        ),
        tag(
            "body",
            tag(
                "header",
                tag("a", "Run viewer", href="/", class_="brand"),
                tag("nav", tag("a", "Batches", href="/"), tag("a", "Compare", href="/compare")),
                tag(
                    "span",
                    "local only: quotes benchmark data and answers",
                    class_="local",
                ),
                class_="top",
            ),
            tag(
                "main",
                tag("div", crumbs, class_="crumbs") if page.crumbs else None,
                tag("h1", page.title),
                page.body,
            ),
        ),
        lang="en",
    )


def error_page(status: int, message: str) -> str:
    return layout(Page(f"{status}", tag("p", message, class_="error-msg"), (("Batches", "/"),)))


__all__ = ["ROUTES", "Context", "NotFound", "Page", "error_page", "esc", "layout"]
