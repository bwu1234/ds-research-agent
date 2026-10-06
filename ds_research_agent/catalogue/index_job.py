"""Controlled indexing job: generate rag-toolkit's config and index the cards.

This is never an agent tool. It runs rag-toolkit's own CLI as a subprocess in
the pinned checkout, so no rag-toolkit internals are imported here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from ds_research_agent.catalogue.models import load_manifest
from ds_research_agent.config import CatalogueSettings, RagServiceSettings

# Metadata carried onto every chunk; these become the filterable fields.
CARRY_METADATA = ["title", "domain", "catalogue_id", "file_path", "format", "card_kind"]
HEADER_TEMPLATE = "{domain} | {format} | {file_path}"

# Run in the service's interpreter; its venv may have no pip.
_LIST_PACKAGES = (
    "import importlib.metadata as m; "
    "print('\\n'.join(sorted({f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions()})))"
)


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: str
    profiler_version: int
    corpus: str
    entry_count: int
    group_count: int
    cards_dir: str
    index_dir: str
    rag_toolkit_revision: str
    rag_config_sha256: str
    rag_packages_sha256: str
    built_at: str
    index_s: float
    index_report: dict[str, Any]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def rag_config(
    base: dict[str, Any], cat: CatalogueSettings, svc: RagServiceSettings
) -> dict[str, Any]:
    """A flattened rag-toolkit config that serves only the card corpus.

    rag-toolkit merges registry mappings from a ``base:`` file, so an overlay
    cannot remove the base corpora. Flattening and replacing ``corpora``
    outright keeps every other corpus out of the served registry.
    """
    if "base" in base:
        raise ValueError("base config itself names a base; flatten it first")
    cards = str(cat.cards_dir.resolve())
    overlay: dict[str, Any] = {
        "paths": {"corpus_dir": cards, "index_dir": str(svc.index_dir.resolve())},
        "chunking": {
            "carry_metadata": CARRY_METADATA,
            "header": {"template": HEADER_TEMPLATE},
        },
        # No retrieval step may call a generative model next to the agent's.
        "retrieval": {"web_search": {"enabled": False}, "expansion": {"provider": "none"}},
        "observability": {"turn_log": {"provider": "none"}},
    }
    merged = _deep_merge(base, overlay)
    merged["corpora"] = {
        "active": [cat.corpus],
        "registry": {
            cat.corpus: {"documents_dir": cards, "description": "Dataset cards (ds-research-agent)"}
        },
    }
    return merged


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def check_checkout(svc: RagServiceSettings) -> None:
    head = _git(svc.checkout, "rev-parse", "HEAD")
    if head != svc.revision:
        raise RuntimeError(f"rag-toolkit checkout is at {head}, pinned {svc.revision}")
    dirty = _git(svc.checkout, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"rag-toolkit checkout has local modifications:\n{dirty}")


def write_rag_config(cat: CatalogueSettings, svc: RagServiceSettings) -> bytes:
    base = yaml.safe_load(svc.base_config.read_text(encoding="utf-8"))
    text = yaml.safe_dump(rag_config(base, cat, svc), sort_keys=True, allow_unicode=True)
    svc.generated_config.parent.mkdir(parents=True, exist_ok=True)
    svc.generated_config.write_text(text, encoding="utf-8")
    return text.encode()


def _rag_cli(svc: RagServiceSettings, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(svc.python), "-m", "rag.cli", "--config", str(svc.generated_config), *args],
        cwd=svc.checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=svc.index_timeout_s,
    )


def _json_report(stdout: str) -> dict[str, Any]:
    # rag-toolkit's CLI logs to stdout even with --json; the report is the
    # trailing object that starts on a line of its own.
    start = 0 if stdout.startswith("{") else stdout.find("\n{\n") + 1
    if start == 0 and not stdout.startswith("{"):
        raise RuntimeError("index-report printed no JSON object")
    report = json.loads(stdout[start:])
    if not isinstance(report, dict):
        raise RuntimeError("index-report JSON is not an object")
    return report


def run_index(cat: CatalogueSettings, svc: RagServiceSettings) -> Generation:
    check_checkout(svc)
    manifest = load_manifest(cat.manifest_path)
    if manifest.corpus != cat.corpus:
        raise RuntimeError(f"manifest is for corpus {manifest.corpus!r}, not {cat.corpus!r}")
    config_bytes = write_rag_config(cat, svc)

    start = time.monotonic()
    _rag_cli(svc, "index", "--corpus", cat.corpus, "--reset")
    index_s = time.monotonic() - start
    report = _json_report(_rag_cli(svc, "index-report", "--corpus", cat.corpus, "--json").stdout)
    if report.get("documents") != len(manifest.entries):
        raise RuntimeError(
            f"index holds {report.get('documents')} documents, manifest {len(manifest.entries)}"
        )
    index = report.get("index", {})
    if index.get("missing") or index.get("stale") or index.get("changed"):
        raise RuntimeError(f"index out of sync with cards: {index}")

    freeze = subprocess.run(
        [str(svc.python), "-c", _LIST_PACKAGES], check=True, capture_output=True, text=True
    ).stdout
    now = dt.datetime.now(dt.UTC)
    gen = Generation(
        generation_id=now.strftime("%Y%m%dT%H%M%SZ"),
        profiler_version=manifest.profiler_version,
        corpus=cat.corpus,
        entry_count=len(manifest.entries),
        group_count=0,
        cards_dir=str(cat.cards_dir.resolve()),
        index_dir=str(svc.index_dir.resolve()),
        rag_toolkit_revision=svc.revision,
        rag_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        rag_packages_sha256=hashlib.sha256(freeze.encode()).hexdigest(),
        built_at=now.isoformat(),
        index_s=round(index_s, 2),
        index_report=report,
    )
    out = svc.index_dir.parent / "generation.json"
    out.write_text(gen.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (svc.index_dir.parent / "rag-packages.txt").write_text(freeze, encoding="utf-8")
    return gen
