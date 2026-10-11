"""Run, replay, score, and report KramaBench baseline batches.

    uv run python -m eval.kramabench.run --config config/local.yaml run \\
        --condition no_tools --tasks smoke [--think low] [--repeats 1]
    uv run python -m eval.kramabench.run --config config/local.yaml replay --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml score --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml report --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml compare --batch A --batch B
    uv run python -m eval.kramabench.run --config config/local.yaml list
    uv run python -m eval.kramabench.run --config config/local.yaml failures --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml show --run RUN_ID
    uv run python -m eval.kramabench.run --config config/local.yaml label --run RUN_ID \\
        --category CATEGORY --note TEXT

Conditions: ``no_tools`` and ``inline`` (D1 baselines, one request per
task), ``given_files`` (the D3 agent), and ``codeagent`` (the D3 minimal
comparator, smolagents CodeAgent on the same sandbox; see
``eval.kramabench.codeagent_runs``). Program conditions need Docker, and
build or reuse the sandbox image. ``run`` calls the local model (slow, free) and needs the
frozen split. Task
sets: ``smoke`` (the two ``-tiny`` tasks), ``sample`` (the fixed D3 sample),
``dev``; ``holdout`` is refused unless ``--unseal-holdout`` is given, which is
for D5 only. ``replay`` re-runs a batch through its recorded responses with
no model or network and fails unless every request hash, answer, and score
matches. Output is aggregates only; per-task answers stay in the ledger,
except ``show``, which renders one run with its gold answer for failure
analysis (local only, like the ledger). ``failures`` applies the rule labels
and lists each failed run's taxonomy label; ``label`` sets one by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from ds_research_agent.agent import DockerWorkspace
from ds_research_agent.config import ModelSettings, Settings, load_settings
from ds_research_agent.ledger import Batch, Ledger, canonical, utc_now
from ds_research_agent.models.ollama_client import OllamaModelClient
from ds_research_agent.sandbox import InputMount, SandboxRunner, SandboxUnavailable
from eval.kramabench import agent_runs, codeagent_runs, failures, report
from eval.kramabench.baselines import (
    CONDITIONS,
    PROGRAM_CONDITIONS,
    SYSTEM_PROMPTS,
    AgentTask,
    Harness,
    ReplayClient,
    run_batch,
    score_run,
    sha256_text,
)
from eval.kramabench.scoring import PROFILE
from eval.kramabench.split import Split, SplitError, frozen_split
from eval.kramabench.tasks import Task

REPO = Path(__file__).resolve().parents[2]
ALL_CONDITIONS = (*CONDITIONS, agent_runs.CONDITION, codeagent_runs.CONDITION)
# Program conditions: the D3 agent and the minimal comparator.
AGENT_TASKS = {
    agent_runs.CONDITION: (agent_runs.run_agent_task, agent_runs.prefix_sha256),
    codeagent_runs.CONDITION: (codeagent_runs.run_codeagent_task, codeagent_runs.prefix_sha256),
}


def _git(*args: str) -> str | None:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def ollama_version(host: str) -> str | None:
    try:
        r = httpx.get(f"{host}/api/version", timeout=5)
        return str(r.json()["version"])
    except httpx.HTTPError, KeyError, ValueError:
        return None


def task_keys(split: Split, name: str, unseal: bool) -> tuple[str, ...]:
    if name == "holdout" and not unseal:
        raise SystemExit("error: the holdout is sealed until D5 (--unseal-holdout)")
    return {
        "smoke": split.smoke,
        "sample": split.sample,
        "dev": split.dev,
        "holdout": split.holdout,
    }[name]


def new_batch(
    settings: Settings,
    model: ModelSettings,
    split: Split,
    *,
    condition: str,
    task_set: str,
    keys: tuple[str, ...],
    repeats: int,
    batch_id: str | None,
    replay_of: Batch | None = None,
    note: str | None = None,
) -> Batch:
    created = utc_now()
    think = "off" if model.think is False else model.think
    stamp = created.replace(":", "").replace("-", "")[:15]
    config = settings.model_dump(mode="json")
    config["model"] = model.model_dump(mode="json")
    dirty = _git("status", "--porcelain")
    return Batch(
        batch_id=batch_id or f"{condition}-{task_set}-{think}-{stamp}",
        created=created,
        condition=condition,
        task_set=task_set,
        task_keys=keys,
        repeats=repeats,
        config=config,
        config_sha256=sha256_text(canonical(config)),
        repo_revision=_git("rev-parse", "HEAD"),
        repo_dirty=None if dirty is None else bool(dirty),
        kramabench_commit=settings.kramabench.commit,
        split_sha256=split.sha256(),
        model_name=model.name,
        model_settings=model.model_dump(mode="json"),
        ollama_version=replay_of.ollama_version if replay_of else ollama_version(model.host),
        answer_type_visible=settings.eval.answer_type_visible,
        scoring_profile=PROFILE,
        system_prompt_sha256=(
            AGENT_TASKS[condition][1]()
            if condition in PROGRAM_CONDITIONS
            else sha256_text(SYSTEM_PROMPTS[condition])  # type: ignore[index]
        ),
        replay_of=replay_of.batch_id if replay_of else None,
        note=note,
    )


def resume_problems(old: Batch, new: Batch) -> list[str]:
    """What differs between a recorded batch and one built from the current
    settings in ways that would change its results."""
    fields = (
        "condition",
        "task_keys",
        "repeats",
        "kramabench_commit",
        "split_sha256",
        "model_name",
        "model_settings",
        "answer_type_visible",
        "scoring_profile",
        "system_prompt_sha256",
    )
    out = [f for f in fields if getattr(old, f) != getattr(new, f)]
    sections = ("eval", "agent", "sandbox") if new.condition in PROGRAM_CONDITIONS else ("eval",)
    for sec in sections:
        if canonical(old.config.get(sec)) != canonical(new.config.get(sec)):
            out.append(f"{sec} settings")
    return out


def docker_agent_task(settings: Settings, condition: str) -> AgentTask:
    """Agent runs on real containers; builds (or reuses) the sandbox image."""
    runner = SandboxRunner(settings.sandbox)
    image = runner.build_image()
    print(f"sandbox image {image.image_id} ({image.strace})", file=sys.stderr)

    def workspace_for(_t: Task, _r: int, mounts: Sequence[InputMount]) -> DockerWorkspace:
        return DockerWorkspace(runner, mounts)

    async def task(h: Harness, b: Batch, t: Task, r: int, client: Any) -> Any:
        run_task = AGENT_TASKS[condition][0]
        return await run_task(h, b, t, r, client, workspace_for, image.image_id)

    return task


def _progress(run: Any, done: int, total: int) -> None:
    print(f"[{done}/{total}] {run.task_key} r{run.repeat}", file=sys.stderr, flush=True)


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    split, tasks = frozen_split(settings)
    ev = settings.eval
    agentic = args.condition in PROGRAM_CONDITIONS
    if not agentic and ev.task_timeout_s > settings.model.request_timeout_s:
        raise SystemExit(
            f"error: eval.task_timeout_s ({ev.task_timeout_s}) exceeds "
            f"model.request_timeout_s ({settings.model.request_timeout_s})"
        )
    model = settings.model
    if model.server_version is not None:
        found = ollama_version(model.host)
        if found != model.server_version:
            raise SystemExit(
                f"error: Ollama at {model.host} reports {found}, but model.server_version "
                f"is {model.server_version}; refusing to mix server versions in results"
            )
    cap = settings.agent.max_output_tokens if agentic else ev.max_output_tokens
    options = model.options | {"num_predict": cap}
    if args.condition == codeagent_runs.CONDITION:
        options["stop"] = codeagent_runs.STOP
    update: dict[str, Any] = {"options": options}
    if args.think is not None:
        update["think"] = False if args.think == "off" else args.think
    model = model.model_copy(update=update)
    keys = task_keys(split, args.tasks, args.unseal_holdout)
    if args.limit:
        keys = keys[: args.limit]
    ledger = Ledger(settings.eval.ledger_path)
    batch = new_batch(
        settings,
        model,
        split,
        condition=args.condition,
        task_set=args.tasks if not args.limit else f"{args.tasks}[:{args.limit}]",
        keys=keys,
        repeats=args.repeats,
        batch_id=args.batch,
        note=args.note,
    )
    existing = {b.batch_id: b for b in ledger.batches()}.get(batch.batch_id)
    if existing is None:
        ledger.add_batch(batch)
    elif not args.resume:
        raise SystemExit(f"error: batch {batch.batch_id} exists; pass --resume to continue it")
    else:
        problems = resume_problems(existing, batch)
        if problems:
            raise SystemExit("error: cannot resume, settings changed: " + "; ".join(problems))
        if existing.repo_revision != batch.repo_revision:
            print(
                f"warning: repository revision changed since the batch started "
                f"({existing.repo_revision} -> {batch.repo_revision})",
                file=sys.stderr,
            )
        dropped = ledger.discard_unfinished(batch.batch_id)
        batch = existing
        done = len(ledger.runs(batch.batch_id))
        print(f"resuming {batch.batch_id}: {done} done, redoing {dropped}", file=sys.stderr)
    print(f"batch {batch.batch_id}: {len(keys)} tasks x {args.repeats}", file=sys.stderr)
    h = Harness(settings, model, ledger, split, tasks)
    client = OllamaModelClient(model)
    agent_task = docker_agent_task(settings, args.condition) if agentic else None
    try:
        asyncio.run(run_batch(h, batch, lambda _t, _r: client, _progress, agent_task))
    except SandboxUnavailable as e:
        print(
            f"error: {e}\nThe current run is left unfinished; once Docker is back, rerun "
            f"with --batch {batch.batch_id} --resume to redo it and continue.",
            file=sys.stderr,
        )
        return 3
    _print_report(ledger, batch, tasks, settings)
    return 0


def cmd_replay(args: argparse.Namespace, settings: Settings) -> int:
    split, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    orig = ledger.batch(args.batch[0])
    model = ModelSettings.model_validate(orig.model_settings)
    if canonical(orig.config.get("eval")) != canonical(settings.eval.model_dump(mode="json")):
        print("warning: eval settings differ from the recorded batch", file=sys.stderr)
    steps = {(r.task_key, r.repeat): ledger.steps(r.run_id) for r in ledger.runs(orig.batch_id)}
    replay = new_batch(
        settings,
        model,
        split,
        condition=orig.condition,
        task_set=orig.task_set,
        keys=orig.task_keys,
        repeats=orig.repeats,
        batch_id=f"{orig.batch_id}-replay-{utc_now()[:19].replace(':', '')}",
        replay_of=orig,
        note="replay of recorded responses",
    )
    ledger.add_batch(replay)
    h = Harness(settings, model, ledger, split, tasks)

    def client_for(t: Task, r: int) -> ReplayClient:
        return ReplayClient(model, steps.get((t.key, r), []))

    agent_task: AgentTask | None = None
    if orig.condition in PROGRAM_CONDITIONS:
        orig_runs = {(r.task_key, r.repeat): r for r in ledger.runs(orig.batch_id)}

        async def agent_task(h: Harness, b: Batch, t: Task, r: int, client: Any) -> Any:
            o = orig_runs[(t.key, r)]
            progs = ledger.programs(o.run_id)
            image = o.input_manifest.get("sandbox_image")
            run_task = AGENT_TASKS[orig.condition][0]
            return await run_task(
                h, b, t, r, client, lambda *_: agent_runs.ReplayWorkspace(progs), image
            )

    asyncio.run(run_batch(h, replay, client_for, agent_task=agent_task))
    diffs = []
    a_runs = {(r.task_key, r.repeat): r for r in ledger.runs(orig.batch_id)}
    for r in ledger.runs(replay.batch_id):
        o = a_runs.get((r.task_key, r.repeat))
        if o is None:
            diffs.append(f"{r.task_key}: not in original")
            continue
        oa, ra = ledger.answer(o.run_id), ledger.answer(r.run_id)
        os_, rs = ledger.score(o.run_id, PROFILE), ledger.score(r.run_id, PROFILE)
        if (oa and oa.model_dump(exclude={"run_id"})) != (ra and ra.model_dump(exclude={"run_id"})):
            diffs.append(f"{r.task_key}: answer differs")
        if (os_ and (os_.score, os_.strict, os_.verified_success)) != (
            rs and (rs.score, rs.strict, rs.verified_success)
        ):
            diffs.append(f"{r.task_key}: score differs")
        ov, rv = ledger.verification(o.run_id), ledger.verification(r.run_id)
        if (ov and ov.model_dump(exclude={"run_id"})) != (rv and rv.model_dump(exclude={"run_id"})):
            diffs.append(f"{r.task_key}: verification differs")
    print(f"replayed {len(a_runs)} runs into {replay.batch_id}: {len(diffs)} differences")
    for d in diffs[:20]:
        print(f"  {d}")
    return 1 if diffs else 0


def cmd_score(args: argparse.Namespace, settings: Settings) -> int:
    _, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    for bid in args.batch:
        runs = ledger.runs(bid)
        for r in runs:
            score_run(ledger, r.run_id, tasks[r.task_key], r.condition)
        print(f"rescored {len(runs)} runs in {bid} with {PROFILE}")
    return 0


def _print_report(ledger: Ledger, batch: Batch, tasks: dict[str, Task], s: Settings) -> None:
    out = report.summarise(ledger, batch, tasks, s.eval.bootstrap_resamples, s.eval.bootstrap_seed)
    print(json.dumps(out, indent=2))


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    _, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    for bid in args.batch:
        _print_report(ledger, ledger.batch(bid), tasks, settings)
    return 0


def cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    split, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    base = ledger.batch(args.batch[0])
    ev = settings.eval
    sample = set(split.sample)
    keys = {
        "all": None,
        "untuned": [k for k in base.task_keys if k not in sample],
        "sample": [k for k in base.task_keys if k in sample],
    }[args.subset]
    for bid in args.batch[1:]:
        out = report.compare(
            ledger,
            base,
            ledger.batch(bid),
            tasks,
            ev.bootstrap_resamples,
            ev.bootstrap_seed,
            keys=keys,
        )
        out["subset"] = args.subset
        print(json.dumps(out, indent=2))
    return 0


def cmd_choose_think(args: argparse.Namespace, settings: Settings) -> int:
    _, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    ev = settings.eval
    sums = [
        report.summarise(ledger, ledger.batch(b), tasks, ev.bootstrap_resamples, ev.bootstrap_seed)
        for b in args.batch
    ]
    out = report.choose_think(sums)
    out["batches"] = {
        s["think"]: {
            "answer_score": s["answer_score"],
            "strict_accuracy": s["strict_accuracy"],
            "median_wall_s": s["wall_s"]["median"],
            "median_output_tokens": (s["output_tokens"] or {}).get("median"),
            "stop_reasons": s["stop_reasons"],
        }
        for s in sums
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_list(args: argparse.Namespace, settings: Settings) -> int:
    ledger = Ledger(settings.eval.ledger_path)
    for b in ledger.batches():
        n = len(ledger.runs(b.batch_id))
        print(f"{b.batch_id}\t{b.condition}\t{b.task_set}\t{n}/{len(b.task_keys) * b.repeats}")
    return 0


def _run(ledger: Ledger, run_id: str) -> Any:
    batch_id = run_id.split("/", 1)[0]
    for r in ledger.runs(batch_id):
        if r.run_id == run_id:
            return r
    raise SystemExit(f"no run {run_id!r}")


def cmd_show(args: argparse.Namespace, settings: Settings) -> int:
    _, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    r = _run(ledger, args.run)
    print(failures.render_run(ledger, r, tasks[r.task_key], args.limit))
    return 0


def cmd_label(args: argparse.Namespace, settings: Settings) -> int:
    ledger = Ledger(settings.eval.ledger_path)
    try:
        lab = failures.label(ledger, _run(ledger, args.run), args.category, args.note)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"{lab.run_id}: {lab.category}")
    return 0


def cmd_failures(args: argparse.Namespace, settings: Settings) -> int:
    _, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    for bid in args.batch:
        n = failures.apply_rules(ledger, bid)
        runs = ledger.runs(bid)
        print(f"{bid}: {n} rule labels applied")
        for r in runs:
            if failures.failed(ledger, r):
                lab = ledger.failure_label(r.run_id)
                what = f"{lab.category} ({lab.source})" if lab else "UNLABELLED"
                print(f"  {r.run_id}\t{r.stop_reason}\t{what}")
        print(json.dumps(failures.counts(ledger, runs, tasks), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kramabench.run")
    ap.add_argument("--config", type=Path, required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    r.add_argument("--tasks", choices=["smoke", "sample", "dev", "holdout"], required=True)
    r.add_argument("--think", choices=["off", "low", "medium", "xhigh"])
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--limit", type=int, help="first N tasks only (timing probes)")
    r.add_argument("--batch", help="batch ID (default: derived from the settings and time)")
    r.add_argument("--note")
    r.add_argument("--unseal-holdout", action="store_true")
    r.add_argument(
        "--resume",
        action="store_true",
        help="create the batch, or continue it if it exists with the same settings",
    )
    for name in ("replay", "score", "report", "compare", "choose-think"):
        p = sub.add_parser(name)
        p.add_argument("--batch", action="append", required=True)
        if name == "compare":
            p.add_argument(
                "--subset",
                choices=("all", "untuned", "sample"),
                default="all",
                help="tasks to pair: all, all but the 12-task tuning sample (the "
                "registered D3 primary result), or the sample alone",
            )
    sub.add_parser("list")
    f = sub.add_parser("failures")
    f.add_argument("--batch", action="append", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("--run", required=True)
    sh.add_argument("--limit", type=int, default=3000, help="characters per message")
    lb = sub.add_parser("label")
    lb.add_argument("--run", required=True)
    lb.add_argument("--category", choices=failures.CATEGORIES, required=True)
    lb.add_argument("--note", required=True)
    args = ap.parse_args(argv)
    settings = load_settings(args.config)
    handlers = {
        "run": cmd_run,
        "replay": cmd_replay,
        "score": cmd_score,
        "report": cmd_report,
        "compare": cmd_compare,
        "choose-think": cmd_choose_think,
        "list": cmd_list,
        "failures": cmd_failures,
        "show": cmd_show,
        "label": cmd_label,
    }
    try:
        return handlers[args.command](args, settings)
    except SplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
