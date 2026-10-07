"""Fetch KramaBench at a pinned commit and split it by visibility.

    uv run python -m eval.kramabench.fetch --config config/local.yaml fetch
    uv run python -m eval.kramabench.fetch --config config/local.yaml verify

``fetch`` clones the configured commit (shallow, by SHA) into an
evaluator-only checkout, then copies:

- upstream ``data/`` into ``visible_root`` (the only agent-visible store, and
  the only input to profiling and the catalogue);
- every other top-level entry except ``dr-input/`` and ``.git`` into
  ``evaluator_root`` (``workload/`` answers, ``solutions/``, scorer code).

Each store gets a ``SHA256SUMS`` file, and ``fetch.json`` records the commit,
tree hash, and counts. Stores are staged and swapped in only when complete.
``verify`` rehashes both stores against their ``SHA256SUMS``. Checksums are
generated locally, not committed: they are benchmark-derived material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ds_research_agent.config import KramaBenchSettings, load_settings
from eval.kramabench.tasks import resolve_sources

VISIBLE_SOURCE = "data"
# Per-task copies of data/ files; cataloguing them would duplicate the pool.
EXCLUDED = {"dr-input", ".git"}
SUMS = "SHA256SUMS"


class FetchError(RuntimeError):
    pass


def git(*args: str, cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise FetchError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def check_layout(s: KramaBenchSettings) -> None:
    """The agent-visible store must not overlap evaluator-only locations."""
    roots = {
        "visible_root": s.visible_root.resolve(),
        "evaluator_root": s.evaluator_root.resolve(),
        "checkout": s.checkout.resolve(),
    }
    names = list(roots)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = roots[a], roots[b]
            if pa == pb or pa.is_relative_to(pb) or pb.is_relative_to(pa):
                raise FetchError(f"{a} ({pa}) and {b} ({pb}) overlap")


def ensure_checkout(s: KramaBenchSettings) -> None:
    co = s.checkout
    if not co.exists():
        co.mkdir(parents=True)
        git("init", "-q", cwd=co)
        git("remote", "add", "origin", s.repo_url, cwd=co)
        print(f"fetching {s.repo_url} @ {s.commit} ...", flush=True)
        git("fetch", "-q", "--depth", "1", "origin", s.commit, cwd=co)
        git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=co)
    head = git("rev-parse", "HEAD", cwd=co)
    if head != s.commit:
        raise FetchError(f"{co} is at {head}, expected {s.commit}")
    dirty = git("status", "--porcelain", "--untracked-files=all", "--ignored", cwd=co)
    if dirty:
        raise FetchError(f"{co} has local changes or extra files:\n{dirty[:2000]}")


def tracked_files(co: Path) -> list[str]:
    """Tracked paths; refuses symlinks and submodules (mode 120000/160000)."""
    out = []
    for line in git("ls-files", "-s", "-z", cwd=co).split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        mode = meta.split()[0]
        if mode not in ("100644", "100755"):
            raise FetchError(f"unsupported tracked entry {path!r} (mode {mode})")
        out.append(path)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_store(co: Path, paths: list[str], strip: str, dest: Path) -> dict[str, Any]:
    """Copy ``paths`` (relative to the checkout) into a staged ``dest``."""
    stage = dest.with_name(f".{dest.name}.staging")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    lines = []
    total = 0
    for rel in sorted(paths):
        out_rel = rel.removeprefix(strip)
        src, dst = co / rel, stage / out_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o444)
        total += dst.stat().st_size
        lines.append(f"{sha256(dst)}  {out_rel}")
    (stage / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if dest.exists():
        # Copies are read-only; make them removable before replacing.
        for p in dest.rglob("*"):
            if p.is_file():
                os.chmod(p, 0o644)
        shutil.rmtree(dest)
    stage.rename(dest)
    return {"files": len(lines), "bytes": total, "sums_sha256": sha256(dest / SUMS)}


def fetch(s: KramaBenchSettings) -> dict[str, Any]:
    check_layout(s)
    start = time.monotonic()
    ensure_checkout(s)
    co = s.checkout
    files = tracked_files(co)
    visible = [p for p in files if p.split("/", 1)[0] == VISIBLE_SOURCE]
    evaluator = [p for p in files if p.split("/", 1)[0] not in EXCLUDED | {VISIBLE_SOURCE}]
    excluded = [p for p in files if p.split("/", 1)[0] in EXCLUDED]
    by_domain = Counter(p.split("/")[1] for p in visible if p.count("/") >= 2)
    record: dict[str, Any] = {
        "repo_url": s.repo_url,
        "commit": s.commit,
        "tree": git("rev-parse", "HEAD^{tree}", cwd=co),
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "visible": copy_store(co, visible, f"{VISIBLE_SOURCE}/", s.visible_root),
        "visible_by_domain": dict(sorted(by_domain.items())),
        "visible_top_level_files": sorted(p for p in visible if p.count("/") == 1),
        "evaluator": copy_store(co, evaluator, "", s.evaluator_root),
        "excluded_files": len(excluded),
    }
    record["data_sources"] = check_sources(s)
    record["seconds"] = round(time.monotonic() - start, 1)
    record_path(s).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check_sources(s: KramaBenchSettings) -> dict[str, Any]:
    """Resolve every task's ``data_sources`` against the visible store.

    Counts entries per tier of :func:`eval.kramabench.tasks.resolve_sources`
    (``exact``, ``nested``, ``casefold``, ``unresolved``) and lists the
    unresolved ones. Counts and paths only, never answers.
    """
    tasks = 0
    tiers: Counter[str] = Counter()
    unresolved: list[str] = []
    for wl in sorted((s.evaluator_root / "workload").glob("*.json")):
        domain = wl.stem.removesuffix("-tiny")
        if not (s.visible_root / domain / "input").is_dir():
            continue  # e.g. quick-start-questions.json
        for task in json.loads(wl.read_text(encoding="utf-8")):
            tasks += 1
            sources = tuple(task.get("data_sources", ()))
            for r in resolve_sources(domain, sources, s.visible_root):
                tiers[r.tier] += 1
                if r.tier == "unresolved":
                    unresolved.append(f"{wl.stem}/{task['id']}: {r.entry}")
    return {"tasks": tasks, "entries": dict(tiers), "unresolved": unresolved}


def record_path(s: KramaBenchSettings) -> Path:
    return s.evaluator_root.parent / "fetch.json"


def verify_store(root: Path) -> list[str]:
    """Problems with ``root`` against its SHA256SUMS; empty when intact."""
    sums = root / SUMS
    if not sums.exists():
        return [f"{sums} missing"]
    expected = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        expected[rel] = digest
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    present.discard(SUMS)
    problems = [f"extra file {p}" for p in sorted(present - expected.keys())]
    for rel, digest in sorted(expected.items()):
        p = root / rel
        if p.is_symlink() or not p.is_file():
            problems.append(f"missing {rel}")
        elif sha256(p) != digest:
            problems.append(f"changed {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kramabench.fetch")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("command", choices=["fetch", "verify"])
    args = ap.parse_args(argv)
    s = load_settings(args.config).kramabench
    try:
        if args.command == "fetch":
            print(json.dumps(fetch(s), indent=2))
        check_layout(s)
        problems = {str(r): verify_store(r) for r in (s.visible_root, s.evaluator_root)}
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    src = check_sources(s)
    print(f"data_sources: {src['tasks']} tasks, entries {src['entries']}")
    for u in src["unresolved"][:20]:
        print(f"  unresolved {u}")
    for root, ps in problems.items():
        print(f"{root}: {'ok' if not ps else f'{len(ps)} problems'}")
        for p in ps[:20]:
            print(f"  {p}")
    return 1 if any(problems.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
