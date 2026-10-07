"""KramaBench tasks from the evaluator-only store, and input resolution.

Tasks are read from ``workload/<name>.json`` in the evaluator store. Each
task is keyed ``<workload>/<id>``: the ``-tiny`` smoke workloads reuse their
parent's task IDs with different fields (``legal-tiny`` changes
``legal-hard-1``'s answer type, sources, and sub-tasks), so a bare ID is not
unique. A task's gold answer lives in :class:`Gold`, separate from the text a
prompt may use, so prompt builders never receive it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Main workloads, one per domain (spelled as on disk).
DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")
# Smoke-test workloads; each task shares an ID with a task in its parent domain.
SMOKE_WORKLOADS = ("environment-tiny", "legal-tiny")
ANSWER_TYPES = (
    "numeric_exact",
    "numeric_approximate",
    "string_exact",
    "string_approximate",
    "list_exact",
    "list_approximate",
)

_ID = re.compile(r"^(?P<domain>[a-z]+)-(?P<difficulty>easy|hard)-(?P<n>\d+)$")

Tier = Literal["exact", "nested", "casefold", "unresolved"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Gold(_Frozen):
    """Evaluator-only. Never passed to a prompt builder or tool."""

    answer: Any
    answer_type: str


class Task(_Frozen):
    key: str  # "<workload>/<id>"
    task_id: str
    workload: str
    domain: str
    difficulty: Literal["easy", "hard"]
    # Key of the main-workload task this belongs to; itself for main tasks.
    parent_key: str
    query: str
    answer_type: str
    data_sources: tuple[str, ...]
    subtask_count: int
    gold: Gold


class TaskSetError(RuntimeError):
    pass


def _parse(workload: str, domain: str, raw: dict[str, Any]) -> Task:
    tid = raw["id"]
    m = _ID.match(tid)
    if not m or m["domain"] != domain:
        raise TaskSetError(f"{workload}: unexpected task id {tid!r}")
    if raw["answer_type"] not in ANSWER_TYPES:
        raise TaskSetError(f"{workload}/{tid}: unknown answer_type {raw['answer_type']!r}")
    return Task(
        key=f"{workload}/{tid}",
        task_id=tid,
        workload=workload,
        domain=domain,
        difficulty=m["difficulty"],  # type: ignore[arg-type]
        parent_key=f"{domain}/{tid}",
        query=raw["query"],
        answer_type=raw["answer_type"],
        data_sources=tuple(raw.get("data_sources", ())),
        subtask_count=len(raw.get("subtasks", ())),
        gold=Gold(answer=raw["answer"], answer_type=raw["answer_type"]),
    )


def load_tasks(evaluator_root: Path) -> dict[str, Task]:
    """Every main and smoke task, keyed by ``Task.key``, in a stable order."""
    tasks: dict[str, Task] = {}
    for workload in (*DOMAINS, *SMOKE_WORKLOADS):
        domain = workload.removesuffix("-tiny")
        path = evaluator_root / "workload" / f"{workload}.json"
        for raw in json.loads(path.read_text(encoding="utf-8")):
            t = _parse(workload, domain, raw)
            if t.key in tasks:
                raise TaskSetError(f"duplicate task {t.key}")
            tasks[t.key] = t
    for t in tasks.values():
        if t.parent_key not in tasks:
            raise TaskSetError(f"{t.key}: parent {t.parent_key} not in a main workload")
    return tasks


def main_tasks(tasks: dict[str, Task]) -> list[Task]:
    return [t for t in tasks.values() if t.workload in DOMAINS]


def workload_identity(evaluator_root: Path) -> dict[str, str]:
    """sha256 of each workload file used, for run records."""
    return {
        w: hashlib.sha256((evaluator_root / "workload" / f"{w}.json").read_bytes()).hexdigest()
        for w in (*DOMAINS, *SMOKE_WORKLOADS)
    }


# --- inputs -------------------------------------------------------------------


class SourceResolution(_Frozen):
    entry: str
    tier: Tier
    # Matched files, relative to the visible root, sorted.
    files: tuple[str, ...]


def _match(base: Path, pattern: str, case_sensitive: bool) -> list[Path]:
    return sorted(p for p in base.glob(pattern, case_sensitive=case_sensitive) if p.is_file())


def resolve_sources(
    domain: str, sources: tuple[str, ...], visible_root: Path
) -> tuple[SourceResolution, ...]:
    """Resolve ``data_sources`` entries against ``<visible_root>/<domain>/input``.

    The first tier that matches wins: ``exact`` (the entry as a glob under
    ``input``), ``nested`` (``**/<entry>``; legal entries omit
    ``csn-data-book-2024-csv/CSVs/``), then ``casefold``, the same two globs
    ignoring case (upstream labels differ in case from files on disk, which a
    Linux sandbox would not forgive). Anything else is ``unresolved``: typos,
    wrong extensions, and files absent from the repository are not guessed at.
    Matching is by string, so macOS's case-insensitive volume does not blur
    the tiers.
    """
    base = visible_root / domain / "input"
    out = []
    for src in sources:
        pattern = src.rstrip("/") + ("/*" if src.endswith("/") else "")
        tiers: tuple[tuple[Tier, str, bool], ...] = (
            ("exact", pattern, True),
            ("nested", f"**/{pattern}", True),
            ("casefold", pattern, False),
            ("casefold", f"**/{pattern}", False),
        )
        res = SourceResolution(entry=src, tier="unresolved", files=())
        for tier, pat, cs in tiers:
            found = _match(base, pat, cs)
            if found:
                files = tuple(str(p.relative_to(visible_root)) for p in found)
                res = SourceResolution(entry=src, tier=tier, files=files)
                break
        out.append(res)
    return tuple(out)


def resolved_files(resolutions: tuple[SourceResolution, ...]) -> tuple[str, ...]:
    """Unique matched files in sorted order."""
    return tuple(sorted({f for r in resolutions for f in r.files}))
