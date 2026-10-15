"""Group-card granularity measurement (offline; synthetic entries)."""

from __future__ import annotations

from ds_research_agent.catalogue.models import CatalogueEntry
from eval.kramabench.grouping import RULES, Schema, partition, selection


def entry(path: str, fmt: str = "csv") -> CatalogueEntry:
    return CatalogueEntry(
        catalogue_id=path,
        domain=path.split("/", 1)[0],
        file_path=path,
        size=10,
        file_hash="sha256:0",
        format=fmt,
        document_id=f"{path}.md",
        card_hash="sha256:0",
        profiler_version=2,
        parse_status="ok",
    )


FILES = [
    "d/a/t-2001.csv",
    "d/a/t-2002.csv",
    "d/a/t-2003.csv",
    "d/a/notes.csv",
    "d/b/t-2004.csv",
    "d/b/single.txt",
]
ENTRIES = [entry(f, f.rsplit(".", 1)[1]) for f in FILES]
TABLE: Schema = (("year", "value"),)
SCHEMAS: dict[str, Schema] = {f: TABLE for f in FILES if "/t-" in f} | {
    "d/a/notes.csv": (("text",),),
    "d/b/single.txt": None,
}
SIZES = {f: 10 for f in FILES}


def test_rules_partition_into_groups_of_two_or_more() -> None:
    assert partition(ENTRIES, SCHEMAS, RULES["directory"]) == [
        ("d/a/notes.csv", "d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv"),
        ("d/b/single.txt", "d/b/t-2004.csv"),
    ]
    family = ("d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv", "d/b/t-2004.csv")
    assert partition(ENTRIES, SCHEMAS, RULES["schema"]) == [family]
    assert partition(ENTRIES, SCHEMAS, RULES["pattern"]) == [family]
    assert partition(ENTRIES, SCHEMAS, RULES["directory+schema"]) == [family[:3]]


def test_policies_count_selections_and_extra_files() -> None:
    family = ("d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv", "d/b/t-2004.csv")
    gold = ["d/a/t-2002.csv", "d/a/t-2003.csv", "d/b/t-2004.csv", "d/b/single.txt"]
    m = selection([family], gold, SIZES)
    # The group is not wholly gold: exact selects its three gold members alone.
    assert m["exact_units"] == 4
    # One contiguous run (2002..2004) plus the ungrouped file.
    assert m["range_units"] == 2
    # The whole family brings t-2001 along.
    assert (m["cover_units"], m["excess"]) == (2, 1)


def test_split_runs_count_separately_and_whole_groups_are_exact() -> None:
    family = ("d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv", "d/b/t-2004.csv")
    gold = ["d/a/t-2001.csv", "d/a/t-2003.csv", "d/b/t-2004.csv"]
    assert selection([family], gold, SIZES)["range_units"] == 2
    whole = selection([family], list(family), SIZES)
    assert (whole["exact_units"], whole["cover_units"], whole["excess"]) == (1, 1, 0)


def test_overlapping_groups_use_the_largest_wholly_gold_ones() -> None:
    by_dir = ("d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv", "d/a/notes.csv")
    family = ("d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv", "d/b/t-2004.csv")
    m = selection(sorted([tuple(sorted(by_dir)), family]), list(family), SIZES)
    assert m["exact_units"] == 1
    # Cover picks, per file, the containing group with the fewest extra files.
    assert (m["cover_units"], m["excess"]) == (1, 0)


def test_pattern_needs_a_digit_run_and_a_shared_schema() -> None:
    entries = [
        entry("d/x/Alabama.csv"),
        entry("d/y/Alabama.csv"),
        entry("d/s/mmc1.xlsx", "xlsx"),
        entry("d/s/mmc2.xlsx", "xlsx"),
        entry("d/o/day1.sp3", "sp3"),
        entry("d/p/day2.sp3", "sp3"),
    ]
    schemas: dict[str, Schema] = {
        "d/x/Alabama.csv": TABLE,
        "d/y/Alabama.csv": TABLE,
        "d/s/mmc1.xlsx": (("id", "age"),),
        "d/s/mmc2.xlsx": (("gene", "score"),),
        "d/o/day1.sp3": None,
        "d/p/day2.sp3": None,
    }
    assert partition(entries, schemas, RULES["pattern"]) == [("d/o/day1.sp3", "d/p/day2.sp3")]
