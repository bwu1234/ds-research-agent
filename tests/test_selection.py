"""Catalogue selections: discovered IDs and member-filter expansion (offline)."""

from __future__ import annotations

import pytest

from ds_research_agent.catalogue import CatalogueEntry, CatalogueGroup, Manifest
from ds_research_agent.catalogue.selection import DiscoveredSet, SelectionItem


def entry(path: str) -> CatalogueEntry:
    return CatalogueEntry(
        catalogue_id=f"id:{path}",
        domain="d",
        file_path=path,
        size=1,
        file_hash="sha256:0",
        format="csv",
        document_id=f"{path}.md",
        card_hash="sha256:0",
        profiler_version=3,
        parse_status="ok",
    )


PATHS = ["d/x/sat-2014.csv", "d/x/sat-2015.csv", "d/x/sat-2016.csv", "d/notes.csv", "d/other.csv"]
GROUP = CatalogueGroup(
    group_id="d/g-1",
    domain="d",
    rules=("pattern",),
    member_ids=tuple(f"id:{p}" for p in PATHS[:3]),
    directory="d/x",
    format="csv",
    document_id="d/_groups/g-1.md",
    card_hash="sha256:0",
    profiler_version=3,
)
MANIFEST = Manifest(
    profiler_version=3, corpus="c", entries=[entry(p) for p in PATHS], skipped=[], groups=[GROUP]
)


def discovered(*ids: str) -> DiscoveredSet:
    d = DiscoveredSet(MANIFEST)
    d.add(ids)
    return d


def test_only_discovered_ids_are_readable_or_selectable() -> None:
    d = discovered("d/g-1", "id:d/notes.csv", "d/g-1")
    assert d.ids == ["d/g-1", "id:d/notes.csv"]
    assert d.record("id:d/other.csv") is None
    r = d.resolve([SelectionItem("id:d/other.csv")])
    assert r.files == () and "not returned by a search" in r.problems[0]
    with pytest.raises(KeyError):
        d.add(["guessed/id"])


def test_a_group_selects_all_members_or_those_matching_a_filter() -> None:
    d = discovered("d/g-1", "id:d/notes.csv")
    whole = d.resolve([SelectionItem("d/g-1")])
    assert [f.file_path for f in whole.files] == PATHS[:3] and whole.problems == ()
    part = d.resolve(
        [SelectionItem("d/g-1", ("*2015*", "*2016*")), SelectionItem("id:d/notes.csv")]
    )
    assert [f.file_path for f in part.files] == [PATHS[1], PATHS[2], PATHS[3]]


def test_filters_never_reach_outside_the_group_and_are_case_sensitive() -> None:
    d = discovered("d/g-1")
    # "d/*" would match d/notes.csv on the filesystem; only members count.
    assert [f.file_path for f in d.resolve([SelectionItem("d/g-1", ("d/*",))]).files] == PATHS[:3]
    r = d.resolve([SelectionItem("d/g-1", ("*SAT*",))])
    assert r.files == () and "matches no member" in r.problems[0]


def test_a_file_takes_no_filter_and_an_empty_selection_is_a_problem() -> None:
    d = discovered("id:d/notes.csv")
    assert (
        "takes no member filter" in d.resolve([SelectionItem("id:d/notes.csv", ("*",))]).problems[0]
    )
    assert d.resolve([]).problems == ("no files were selected",)
