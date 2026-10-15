"""Catalogue selections: which files a run may use, by catalogue or group ID.

An agent selects only IDs its run has *discovered* (returned by search, or
given by the condition), never paths. A group ID selects its listed members,
or those matching a member filter: globs over member paths, ``fnmatch`` style
and case-sensitive (see docs/data-and-provenance.md#group-cards). Filters are
expanded against the group's member list from the manifest, never against
the filesystem, so a glob cannot reach a file outside the group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

from ds_research_agent.catalogue.models import CatalogueEntry, CatalogueGroup, Manifest


@dataclass(frozen=True)
class SelectionItem:
    id: str
    # Globs over member paths; only for a group. Empty selects every member.
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Resolved:
    items: tuple[SelectionItem, ...]
    # Selected files, in manifest order, without duplicates.
    files: tuple[CatalogueEntry, ...]
    problems: tuple[str, ...]


@dataclass
class DiscoveredSet:
    """The catalogue and group IDs a run may read or select, in first-seen order."""

    manifest: Manifest
    ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._entries = self.manifest.by_catalogue_id()
        self._groups = {g.group_id: g for g in self.manifest.groups}

    def add(self, ids: Iterable[str]) -> None:
        for i in ids:
            if i not in self._entries and i not in self._groups:
                raise KeyError(f"{i!r} is not in the manifest")
            if i not in self.ids:
                self.ids.append(i)

    def __contains__(self, i: object) -> bool:
        return i in self.ids

    def record(self, i: str) -> CatalogueEntry | CatalogueGroup | None:
        """The discovered entry or group, or None if ``i`` was not discovered."""
        if i not in self.ids:
            return None
        return self._entries.get(i) or self._groups[i]

    def members(self, group: CatalogueGroup, globs: Sequence[str] = ()) -> list[CatalogueEntry]:
        members = [self._entries[m] for m in group.member_ids]
        if not globs:
            return members
        return [m for m in members if any(fnmatchcase(m.file_path, g) for g in globs)]

    def resolve(self, items: Sequence[SelectionItem]) -> Resolved:
        """Expand a selection; every problem is reported, nothing is guessed."""
        problems: list[str] = []
        chosen: set[str] = set()
        for item in items:
            rec = self.record(item.id)
            if rec is None:
                problems.append(f"{item.id!r} was not returned by a search in this run")
            elif isinstance(rec, CatalogueEntry):
                if item.members:
                    problems.append(f"{item.id!r} is a file, so it takes no member filter")
                else:
                    chosen.add(rec.catalogue_id)
            else:
                matched = self.members(rec, item.members)
                if not matched:
                    problems.append(
                        f"member filter {list(item.members)} matches no member of {item.id!r}"
                    )
                chosen.update(m.catalogue_id for m in matched)
        files = tuple(e for e in self.manifest.entries if e.catalogue_id in chosen)
        if not items:
            problems.append("no files were selected")
        return Resolved(items=tuple(items), files=files, problems=tuple(problems))
