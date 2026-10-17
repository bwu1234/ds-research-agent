"""Run, replay, score, and report KramaBench baseline batches.

    uv run python -m eval.kramabench.run --config config/local.yaml run \\
        --condition no_tools --tasks smoke [--think low] [--repeats 1]
    uv run python -m eval.kramabench.run --config config/local.yaml replay --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml score --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml report --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml compare --batch A --batch B
    uv run python -m eval.kramabench.run --config config/local.yaml compare \\
        --batch A1 --batch A2 --vs B1 --vs B2 --subset untuned [--rule d4]
    uv run python -m eval.kramabench.run --config config/local.yaml list
    uv run python -m eval.kramabench.run --config config/local.yaml failures --batch ID
    uv run python -m eval.kramabench.run --config config/local.yaml show --run RUN_ID
    uv run python -m eval.kramabench.run --config config/local.yaml label --run RUN_ID \\
        --category CATEGORY --note TEXT
    uv run python -m eval.kramabench.run --config config/local.yaml serve [--port 8765]

Conditions: ``no_tools`` and ``inline`` (D1 baselines, one request per
task), ``given_files`` (the D3 agent), ``codeagent`` (the D3 minimal
comparator, smolagents CodeAgent on the same sandbox; see
``eval.kramabench.codeagent_runs``), ``discovery_agent`` (D2; chooses files,
no answer), and ``end_to_end`` (D4: single-shot search, then the analysis
agent on the files it selects; needs the rag-toolkit server and the
``end_to_end`` settings). Program conditions need Docker, and
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
``serve`` starts the run viewer (``eval.kramabench.viewer``) on 127.0.0.1: the
ledger opened read-only, as pages for batches, runs with their traces, tasks,
and comparisons. Like ``show`` it quotes gold answers, so it stays local; holdout
batches and runs stay hidden unless ``--unseal-holdout`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from ds_research_agent.agent import DockerWorkspace
from ds_research_agent.catalogue import load_manifest
from ds_research_agent.config import ModelSettings, Settings, load_settings
from ds_research_agent.ledger import Batch, Ledger, canonical, utc_now
from ds_research_agent.models.ollama_client import OllamaModelClient
from ds_research_agent.retrieval.mcp_adapter import McpRetrieval
from ds_research_agent.sandbox import InputMount, SandboxRunner, SandboxUnavailable
from eval.kramabench import (
    agent_runs,
    codeagent_runs,
    discovery_agent,
    end_to_end,
    failures,
    report,
    viewer,
)
from eval.kramabench.baselines import (
    AGENT_CONDITIONS,
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
ALL_CONDITIONS = (
    *CONDITIONS,
    agent_runs.CONDITION,
    codeagent_runs.CONDITION,
    discovery_agent.CONDITION,
    end_to_end.CONDITION,
)
# Given-files program conditions: the D3 agent and the minimal comparator.
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
        version = r.json()["version"]
    except httpx.HTTPError, KeyError, TypeError, ValueError:
        return None
    return version if isinstance(version, str) and version.strip() else None


def server_version_problem(found: str | None, refused: Sequence[str]) -> str | None:
    """Why a batch may not start on this server, or None. A version off the
    refusal list runs; it is recorded on the batch and compared by ``compare``."""
    if found is None:
        return "cannot read /api/version, so the server version cannot be recorded"
    if found in refused:
        return f"version {found} is in model.refuse_server_versions (known to fail)"
    return None


class ServerVersionChanged(RuntimeError):
    """The Ollama server's version no longer matches the batch's."""


def version_guard(host: str, recorded: str | None) -> Callable[[], None]:
    """A ``before_run`` check: the server must still report the batch's version.
    Auto-update installs a new version when the Ollama app restarts, which can
    happen mid-batch; the batch is stamped once, so later runs must not drift."""

    def check() -> None:
        found = ollama_version(host)
        if found != recorded:
            raise ServerVersionChanged(
                f"Ollama at {host} reports {found or 'no version (unreachable)'}, "
                f"but the batch was recorded on {recorded}"
            )

    return check


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
        scoring_profile=profile_for(condition),
        system_prompt_sha256=(
            discovery_agent.prefix_sha256()
            if condition == discovery_agent.CONDITION
            else end_to_end.prefix_sha256()
            if condition == end_to_end.CONDITION
            else AGENT_TASKS[condition][1]()
            if condition in PROGRAM_CONDITIONS
            else sha256_text(SYSTEM_PROMPTS[condition])  # type: ignore[index]
        ),
        replay_of=replay_of.batch_id if replay_of else None,
        note=note,
    )


def profile_for(condition: str) -> str:
    return discovery_agent.PROFILE if condition == discovery_agent.CONDITION else PROFILE


def batch_sections(condition: str) -> tuple[str, ...]:
    """Settings sections that can change a batch's results."""
    if condition == discovery_agent.CONDITION:
        return ("eval", "agent", "discovery", "catalogue", "rag_service")
    if condition == end_to_end.CONDITION:
        return (
            "eval", "agent", "sandbox", "discovery", "end_to_end", "catalogue", "rag_service"
        )  # fmt: skip
    return ("eval", "agent", "sandbox") if condition in PROGRAM_CONDITIONS else ("eval",)


def comparable_model_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Model settings that can change a result. The legacy exact pin and the
    refusal list only gate a start; the version itself is compared apart."""
    return {
        k: v for k, v in settings.items() if k not in ("server_version", "refuse_server_versions")
    }


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
    out = [f for f in fields if f != "model_settings" and getattr(old, f) != getattr(new, f)]
    if comparable_model_settings(old.model_settings) != comparable_model_settings(
        new.model_settings
    ):
        out.append("model_settings")
    if old.ollama_version is None:
        out.append("Ollama version (not recorded on the batch; start a new batch)")
    elif old.ollama_version != new.ollama_version:
        out.append(f"Ollama version ({old.ollama_version} -> {new.ollama_version})")
    for sec in batch_sections(new.condition):
        if canonical(old.config.get(sec)) != canonical(new.config.get(sec)):
            out.append(f"{sec} settings")
    return out


def sandbox_image(settings: Settings) -> tuple[SandboxRunner, str]:
    """The runner, and the image it builds (or reuses) for this batch."""
    runner = SandboxRunner(settings.sandbox)
    image = runner.build_image()
    print(f"sandbox image {image.image_id} ({image.strace})", file=sys.stderr)
    return runner, image.image_id


def docker_agent_task(settings: Settings, condition: str) -> AgentTask:
    """Agent runs on real containers; builds (or reuses) the sandbox image."""
    runner, image_id = sandbox_image(settings)

    def workspace_for(_t: Task, _r: int, mounts: Sequence[InputMount]) -> DockerWorkspace:
        return DockerWorkspace(runner, mounts)

    async def task(h: Harness, b: Batch, t: Task, r: int, client: Any) -> Any:
        run_task = AGENT_TASKS[condition][0]
        return await run_task(h, b, t, r, client, workspace_for, image_id)

    return task


def discovery_manifest(settings: Settings) -> Any:
    manifest = load_manifest(settings.catalogue.manifest_path)
    if manifest.corpus != settings.catalogue.corpus:
        raise SystemExit(
            f"error: manifest is for corpus {manifest.corpus!r}, not {settings.catalogue.corpus!r}"
        )
    return manifest


async def _run_discovery_batch(
    h: Harness, batch: Batch, client: Any, guard: Callable[[], None]
) -> None:
    """One MCP server for the whole batch; each run records its searches."""
    manifest = discovery_manifest(h.settings)
    log = h.settings.eval.ledger_path.parent / f"{batch.batch_id}.server.log"
    with log.open("a", encoding="utf-8") as errlog:
        async with McpRetrieval(h.settings.mcp, errlog=errlog) as r:

            async def task(h: Harness, b: Batch, t: Task, rep: int, c: Any) -> Any:
                return await discovery_agent.run_discovery_task(h, b, t, rep, c, r, manifest)

            await run_batch(h, batch, lambda _t, _r: client, _progress, task, guard)


async def _run_end_to_end_batch(
    h: Harness, batch: Batch, client: Any, guard: Callable[[], None]
) -> None:
    """One MCP server for the batch, and one workspace per run, starting with
    nothing mounted; the agent's selections set the mounts."""
    manifest = discovery_manifest(h.settings)
    runner, image_id = sandbox_image(h.settings)
    log = h.settings.eval.ledger_path.parent / f"{batch.batch_id}.server.log"
    with log.open("a", encoding="utf-8") as errlog:
        async with McpRetrieval(h.settings.mcp, errlog=errlog) as r:

            async def task(h: Harness, b: Batch, t: Task, rep: int, c: Any) -> Any:
                return await end_to_end.run_end_to_end_task(
                    h, b, t, rep, c, lambda *_: DockerWorkspace(runner, []), image_id, r, manifest
                )

            await run_batch(h, batch, lambda _t, _r: client, _progress, task, guard)


def _progress(run: Any, done: int, total: int) -> None:
    print(f"[{done}/{total}] {run.task_key} r{run.repeat}", file=sys.stderr, flush=True)


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    split, tasks = frozen_split(settings)
    ev = settings.eval
    if args.condition == end_to_end.CONDITION and settings.end_to_end is None:
        raise SystemExit("error: the end_to_end condition needs an end_to_end settings section")
    agentic = args.condition in AGENT_CONDITIONS
    if not agentic and ev.task_timeout_s > settings.model.request_timeout_s:
        raise SystemExit(
            f"error: eval.task_timeout_s ({ev.task_timeout_s}) exceeds "
            f"model.request_timeout_s ({settings.model.request_timeout_s})"
        )
    model = settings.model
    problem = server_version_problem(ollama_version(model.host), model.refuse_server_versions)
    if problem:
        raise SystemExit(f"error: Ollama at {model.host}: {problem}")
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
    h = Harness(settings, model, ledger, split, tasks, repo_revision=_git("rev-parse", "HEAD"))
    client = OllamaModelClient(model)
    guard = version_guard(model.host, batch.ollama_version)
    try:
        if args.condition == discovery_agent.CONDITION:
            asyncio.run(_run_discovery_batch(h, batch, client, guard))
        elif args.condition == end_to_end.CONDITION:
            asyncio.run(_run_end_to_end_batch(h, batch, client, guard))
        else:
            agent_task = docker_agent_task(settings, args.condition) if agentic else None
            asyncio.run(run_batch(h, batch, lambda _t, _r: client, _progress, agent_task, guard))
    except ServerVersionChanged as e:
        msg = (
            f"error: {e}. The batch stopped before its next run and cannot be "
            "resumed on another version; start a new batch."
        )
        if last := ledger.runs(batch.batch_id)[-1:]:
            msg += (
                f" The last finished run ({last[0].task_key}, stop reason "
                f"{last[0].stop_reason}) may have spanned the restart."
            )
        print(msg, file=sys.stderr)
        return 4
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
    orig_runs = {(r.task_key, r.repeat): r for r in ledger.runs(orig.batch_id)}
    all_steps = {k: ledger.steps(r.run_id) for k, r in orig_runs.items()}
    steps = {k: [s for s in v if s.kind == "model"] for k, v in all_steps.items()}
    # Replay cannot reach a wall-clock limit; these runs replay theirs from the recording.
    at_deadline = {k for k, r in orig_runs.items() if r.stop_reason == "max_wall"}
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
    h = Harness(settings, model, ledger, split, tasks, repo_revision=_git("rev-parse", "HEAD"))

    def client_for(t: Task, r: int) -> ReplayClient:
        return ReplayClient(model, steps.get((t.key, r), []), (t.key, r) in at_deadline)

    agent_task: AgentTask | None = None
    if orig.condition == discovery_agent.CONDITION:
        manifest = discovery_manifest(settings)

        async def agent_task(h: Harness, b: Batch, t: Task, r: int, client: Any) -> Any:
            searcher = discovery_agent.ReplaySearcher(all_steps.get((t.key, r), []))
            return await discovery_agent.run_discovery_task(h, b, t, r, client, searcher, manifest)

    elif orig.condition == end_to_end.CONDITION:
        manifest = discovery_manifest(settings)

        async def agent_task(h: Harness, b: Batch, t: Task, r: int, client: Any) -> Any:
            o = orig_runs[(t.key, r)]
            searcher = discovery_agent.ReplaySearcher(all_steps.get((t.key, r), []))
            ws = agent_runs.ReplayWorkspace(ledger.programs(o.run_id), (t.key, r) in at_deadline)
            image = o.input_manifest.get("sandbox_image")
            return await end_to_end.run_end_to_end_task(
                h, b, t, r, client, lambda *_: ws, image, searcher, manifest
            )

    elif orig.condition in PROGRAM_CONDITIONS:

        async def agent_task(h: Harness, b: Batch, t: Task, r: int, client: Any) -> Any:
            o = orig_runs[(t.key, r)]
            progs = ledger.programs(o.run_id)
            image = o.input_manifest.get("sandbox_image")
            run_task = AGENT_TASKS[orig.condition][0]
            ws = agent_runs.ReplayWorkspace(progs, (t.key, r) in at_deadline)
            return await run_task(h, b, t, r, client, lambda *_: ws, image)

    asyncio.run(run_batch(h, replay, client_for, agent_task=agent_task))
    diffs = []
    a_runs = {(r.task_key, r.repeat): r for r in ledger.runs(orig.batch_id)}
    for r in ledger.runs(replay.batch_id):
        o = a_runs.get((r.task_key, r.repeat))
        if o is None:
            diffs.append(f"{r.task_key}: not in original")
            continue
        oa, ra = ledger.answer(o.run_id), ledger.answer(r.run_id)
        prof = profile_for(orig.condition)
        os_, rs = ledger.score(o.run_id, prof), ledger.score(r.run_id, prof)
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
        cond = ledger.batch(bid).condition
        for r in runs:
            if cond == discovery_agent.CONDITION:
                root = settings.kramabench.visible_root
                discovery_agent.score_discovery(ledger, r.run_id, tasks[r.task_key], root)
            else:
                score_run(ledger, r.run_id, tasks[r.task_key], r.condition)
        print(f"rescored {len(runs)} runs in {bid} with {profile_for(cond)}")
    return 0


def _print_report(
    ledger: Ledger, batch: Batch, tasks: dict[str, Task], s: Settings, out_path: Path | None = None
) -> None:
    if batch.condition == discovery_agent.CONDITION:
        # Discovery-format output, so eval.kramabench.discovery_paired can pair it
        # with single-shot results; rows hold gold-set sizes, not answers.
        out = discovery_agent.summary(ledger, batch, tasks, s.kramabench.visible_root)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out["summary"], indent=2))
        return
    out = report.summarise(ledger, batch, tasks, s.eval.bootstrap_resamples, s.eval.bootstrap_seed)
    print(json.dumps(out, indent=2))


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    _, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    if args.out is not None and len(args.batch) != 1:
        raise SystemExit("error: --out takes one --batch")
    for bid in args.batch:
        _print_report(ledger, ledger.batch(bid), tasks, settings, args.out)
    return 0


def cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    split, tasks = frozen_split(settings)
    ledger = Ledger(settings.eval.ledger_path)
    base = ledger.batch(args.batch[0])
    if any(ledger.batch(b).condition == discovery_agent.CONDITION for b in args.batch + args.vs):
        raise SystemExit(
            "error: compare pairs answer scores; for discovery_agent batches, write "
            "`report --out` files and pair them with eval.kramabench.discovery_paired"
        )
    ev = settings.eval
    sample = set(split.sample)
    keys = {
        "all": None,
        "untuned": [k for k in base.task_keys if k not in sample],
        "sample": [k for k in base.task_keys if k in sample],
    }[args.subset]
    if args.vs:
        return _compare_pooled(args, settings, ledger, tasks, keys)
    if args.rule:
        raise SystemExit("error: --rule applies to a pooled comparison (--vs)")
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
        other = ledger.batch(bid)
        if base.ollama_version is None or base.ollama_version != other.ollama_version:
            out["warning"] = (
                f"Ollama versions differ or are unknown: {base.ollama_version} "
                f"({base.batch_id}) vs {other.ollama_version} ({other.batch_id})"
            )
            print(f"warning: {out['warning']}", file=sys.stderr)
        print(json.dumps(out, indent=2))
    return 0


def _compare_pooled(
    args: argparse.Namespace,
    settings: Settings,
    ledger: Ledger,
    tasks: dict[str, Task],
    keys: list[str] | None,
) -> int:
    """Every --batch (passes of the base condition) against every --vs (passes
    of the other), pooled per task; with --rule d4, the registered decision."""
    base = [ledger.batch(b) for b in args.batch]
    other = [ledger.batch(b) for b in args.vs]
    for side, name in ((base, "--batch"), (other, "--vs")):
        if len({b.condition for b in side}) != 1:
            raise SystemExit(f"error: the {name} batches must share a condition")
    versions = {b.ollama_version for b in base + other}
    if len(versions) != 1 or None in versions:
        raise SystemExit(f"error: a pooled comparison needs one known Ollama version: {versions}")
    ev = settings.eval
    out = report.compare(
        ledger, base, other, tasks, ev.bootstrap_resamples, ev.bootstrap_seed, keys=keys
    )
    out["subset"] = args.subset
    if args.rule == "d4":
        if (
            (base[0].condition, other[0].condition)
            != (
                agent_runs.CONDITION,
                end_to_end.CONDITION,
            )
            or args.subset != "untuned"
            or len(base) != 2
            or len(other) != 2
        ):
            raise SystemExit(
                "error: the D4 rule is registered for given_files (--batch, 2 passes) against "
                "end_to_end (--vs, 2 passes) on --subset untuned"
            )
        out["d4"] = report.d4_decision(out)
    if other[0].condition == end_to_end.CONDITION:
        # Diagnostic, not part of the registered rule.
        out["by_selection"] = report.by_selection(
            ledger, base, other, tasks, ev.bootstrap_resamples, ev.bootstrap_seed, keys=keys
        )
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
    sel = report.run_selection(ledger, r.run_id, tasks) if r.condition == "end_to_end" else None
    print(failures.render_run(ledger, r, tasks[r.task_key], args.limit, sel))
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


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    split, tasks = frozen_split(settings)
    ctx = viewer.Context(settings, split, tasks, unseal_holdout=args.unseal_holdout)
    return viewer.serve(ctx, settings.eval.ledger_path, args.port)


def cmd_backfill_revision(args: argparse.Namespace, settings: Settings) -> int:
    """Set repo_revision on runs recorded before schema 4 (never overwrites)."""
    ledger = Ledger(settings.eval.ledger_path)
    ledger.batch(args.batch)  # unknown batch is an error
    n = ledger.set_run_revision(
        args.batch,
        args.revision,
        started_from=args.started_from,
        started_before=args.started_before,
    )
    print(f"{args.batch}: set repo_revision {args.revision} on {n} runs")
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
        if name == "report":
            p.add_argument("--out", type=Path, help="discovery_agent: write the rows here")
        if name == "compare":
            p.add_argument(
                "--subset",
                choices=("all", "untuned", "sample"),
                default="all",
                help="tasks to pair: all, all but the 12-task tuning sample (the "
                "registered D3 primary result), or the sample alone",
            )
            p.add_argument(
                "--vs",
                action="append",
                default=[],
                help="other-condition batch; with --vs, every --batch is pooled against "
                "every --vs (per-task mean over passes) in one comparison",
            )
            p.add_argument(
                "--rule",
                choices=("d4",),
                help="apply a registered decision rule to the pooled comparison",
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
    sv = sub.add_parser("serve", help="local read-only viewer for the ledger")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--unseal-holdout", action="store_true")
    bf = sub.add_parser("backfill-revision")
    bf.add_argument("--batch", required=True)
    bf.add_argument("--revision", required=True)
    bf.add_argument("--started-from", help="only runs started at or after this ISO time")
    bf.add_argument("--started-before", help="only runs started before this ISO time")
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
        "serve": cmd_serve,
        "backfill-revision": cmd_backfill_revision,
    }
    try:
        return handlers[args.command](args, settings)
    except SplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
