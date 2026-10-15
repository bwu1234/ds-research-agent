"""Group-card granularity measurement (offline; synthetic entries)."""

from __future__ import annotations

from ds_research_agent.catalogue.grouping import (
    Schema,
    common_directory,
    families,
    heading,
    name_pattern,
    partition,
)
from ds_research_agent.catalogue.models import CatalogueEntry
from eval.kramabench.grouping import selection


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
    assert partition(ENTRIES, SCHEMAS, "directory") == [
        ("d/a/notes.csv", "d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv"),
        ("d/b/single.txt", "d/b/t-2004.csv"),
    ]
    family = ("d/a/t-2001.csv", "d/a/t-2002.csv", "d/a/t-2003.csv", "d/b/t-2004.csv")
    assert partition(ENTRIES, SCHEMAS, "schema") == [family]
    assert partition(ENTRIES, SCHEMAS, "pattern") == [family]
    assert partition(ENTRIES, SCHEMAS, "directory+schema") == [family[:3]]


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
    assert partition(entries, schemas, "pattern") == [("d/o/day1.sp3", "d/p/day2.sp3")]


def test_identical_member_sets_from_two_rules_are_one_group() -> None:
    fams = families(ENTRIES, SCHEMAS, ["schema", "pattern"])
    assert len(fams) == 1
    (fam,) = fams
    assert fam.rules == ("pattern", "schema") and fam.domain == "d"
    assert fam.directory == "d" and fam.group_id.startswith("d/g-")
    # The ID depends only on the members, so it is stable across builds.
    assert families(ENTRIES, SCHEMAS, ["pattern"])[0].group_id == fam.group_id


def test_name_pattern_and_common_directory() -> None:
    names = ["sp3-wu002-20140131.sp3", "sp3-wu010-20140203.sp3", "sp3-wu001-20140128.sp3"]
    # Constant runs stay literal; varying ones are masked, with their range.
    assert name_pattern(names) == ("sp3-wu#-#.sp3", [("001", "010"), ("20140128", "20140203")])
    assert name_pattern(["a1.csv", "b2.csv"]) is None
    assert common_directory(["d/x/1/a", "d/x/2/b"]) == "d/x"


def test_heading_names_a_family_by_pattern_affixes_or_directory() -> None:
    assert heading(["t-2001.csv", "t-2002.csv"], "d/a") == "t-#.csv"
    beaches = ["city_point_beach_datasheet.csv", "malibu_beach_datasheet.csv"]
    assert heading(beaches, "env/input") == "*_beach_datasheet.csv"
    assert heading(["Alabama.csv", "Wyoming.csv"], "legal/State data") == "State data/*.csv"
