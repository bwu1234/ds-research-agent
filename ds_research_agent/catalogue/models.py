"""Catalogue records: one entry per raw file and one per file family (group),
written by the profiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

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
    # Groups this file belongs to (groups may overlap), sorted.
    group_ids: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _legacy_group_id(cls, data: Any) -> Any:
        # Version 1 and 2 manifests carry ``group_id: null``.
        if isinstance(data, dict) and "group_id" in data:
            data = dict(data)
            if data.pop("group_id") is not None:
                raise ValueError("group_id is replaced by group_ids")
        return data


class CatalogueGroup(BaseModel):
    """A file family with its own card; see docs/data-and-provenance.md#group-cards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    domain: str
    # Rules that produced exactly this member set, sorted.
    rules: tuple[str, ...]
    # Member catalogue IDs, in member path order.
    member_ids: tuple[str, ...]
    # The members' common directory, relative to the data root.
    directory: str
    format: str
    document_id: str
    card_hash: str
    profiler_version: int


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
    groups: list[CatalogueGroup] = []

    def by_document_id(self) -> dict[str, CatalogueEntry]:
        return {e.document_id: e for e in self.entries}

    def by_catalogue_id(self) -> dict[str, CatalogueEntry]:
        return {e.catalogue_id: e for e in self.entries}

    def groups_by_document_id(self) -> dict[str, CatalogueGroup]:
        return {g.document_id: g for g in self.groups}

    def members(self, group: CatalogueGroup) -> list[CatalogueEntry]:
        by_id = self.by_catalogue_id()
        return [by_id[i] for i in group.member_ids]


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
