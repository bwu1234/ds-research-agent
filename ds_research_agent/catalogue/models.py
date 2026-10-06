"""Catalogue records: one entry per raw file, written by the profiler."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

ParseStatus = Literal["ok", "unsupported", "error"]


class CatalogueEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalogue_id: str
    domain: str
    # Relative to the data root, POSIX separators.
    file_path: str
    size: int
    file_hash: str
    format: str
    # The card's path relative to the cards directory. rag-toolkit's Markdown
    # loader uses exactly this as the document ID.
    document_id: str
    card_hash: str
    profiler_version: int
    parse_status: ParseStatus
    parse_error: str | None = None
    group_id: str | None = None


class SkippedPath(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    reason: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profiler_version: int
    corpus: str
    entries: list[CatalogueEntry]
    skipped: list[SkippedPath]

    def by_document_id(self) -> dict[str, CatalogueEntry]:
        return {e.document_id: e for e in self.entries}

    def by_catalogue_id(self) -> dict[str, CatalogueEntry]:
        return {e.catalogue_id: e for e in self.entries}


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
