"""File families for group cards (see docs/data-and-provenance.md#group-cards).

A rule maps a catalogue entry to a key; entries of one domain with the same
key form a family when there are two or more. Several rules may yield the same
member set, which is then one group recording each rule. Everything here is
deterministic: the same entries, schemas, and rules give the same groups.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from ds_research_agent.catalogue.models import CatalogueEntry
from ds_research_agent.config.settings import GroupRule

# Column names of every table the profiler finds, in order; None when it
# finds no table with columns.
Schema = tuple[tuple[str, ...], ...] | None
Key = tuple[str, ...]

DIGITS = re.compile(r"\d+")


def _directory(e: CatalogueEntry, _: Schema) -> Key:
    return (str(PurePosixPath(e.file_path).parent),)


def _schema(e: CatalogueEntry, schema: Schema) -> Key | None:
    if schema is None:
        return None
    return (e.domain, e.format, json.dumps(schema))


def _directory_schema(e: CatalogueEntry, schema: Schema) -> Key | None:
    key = _schema(e, schema)
    return None if key is None else _directory(e, schema) + key


def _pattern(e: CatalogueEntry, schema: Schema) -> Key | None:
    # A name with no digit run is not a numbered family (``Alabama.csv`` in two
    # directories), and members must share the schema when there is one
    # (``mmc1.xlsx`` .. ``mmc7.xlsx`` with different tables are not one table).
    name = PurePosixPath(e.file_path).name
    masked = mask(name)
    if masked == name:
        return None
    return (e.domain, masked, json.dumps(schema))


RULES: dict[GroupRule, Callable[[CatalogueEntry, Schema], Key | None]] = {
    "directory": _directory,
    "schema": _schema,
    "directory+schema": _directory_schema,
    "pattern": _pattern,
}


def mask(name: str) -> str:
    return DIGITS.sub("#", name)


def partition(
    entries: Iterable[CatalogueEntry], schemas: Mapping[str, Schema], rule: GroupRule
) -> list[tuple[str, ...]]:
    """One rule's families: sorted member paths, two or more, in a stable order."""
    key_of = RULES[rule]
    by: dict[Key, list[str]] = defaultdict(list)
    for e in entries:
        key = key_of(e, schemas[e.file_path])
        if key is not None:
            by[(e.domain, *key)].append(e.file_path)
    return sorted(tuple(sorted(fs)) for fs in by.values() if len(fs) > 1)


@dataclass(frozen=True)
class Family:
    domain: str
    members: tuple[str, ...]
    rules: tuple[str, ...]

    @property
    def group_id(self) -> str:
        digest = hashlib.sha256("\n".join(self.members).encode()).hexdigest()[:10]
        return f"{self.domain}/g-{digest}"

    @property
    def directory(self) -> str:
        return common_directory(self.members)


def families(
    entries: Sequence[CatalogueEntry], schemas: Mapping[str, Schema], rules: Sequence[GroupRule]
) -> list[Family]:
    """Every configured rule's families, identical member sets merged."""
    domain = {e.file_path: e.domain for e in entries}
    by_members: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for rule in rules:
        for members in partition(entries, schemas, rule):
            by_members[members].add(rule)
    return [
        Family(domain=domain[m[0]], members=m, rules=tuple(sorted(r)))
        for m, r in sorted(by_members.items())
    ]


def common_directory(paths: Sequence[str]) -> str:
    parts = [PurePosixPath(p).parent.parts for p in paths]
    out: list[str] = []
    for level in zip(*parts, strict=False):
        if len(set(level)) != 1:
            break
        out.append(level[0])
    return "/".join(out)


def name_pattern(names: Sequence[str]) -> tuple[str, list[tuple[str, str]]] | None:
    """The shared name with varying digit runs as ``#``, and their first and
    last values; None unless every name has the same masked form.

    Runs that are the same in every name stay literal (``SP3ACOM``, ``.sp3``).
    """
    masked = {mask(n) for n in names}
    if len(masked) != 1:
        return None
    runs = [DIGITS.findall(n) for n in names]
    columns = list(zip(*runs, strict=True))
    pending = iter(columns)

    def run(_: re.Match[str]) -> str:
        values = next(pending)
        return values[0] if len(set(values)) == 1 else "#"

    shown = DIGITS.sub(run, names[0])
    ranges = [
        (min(c, key=lambda v: (int(v), v)), max(c, key=lambda v: (int(v), v)))
        for c in columns
        if len(set(c)) > 1
    ]
    return shown, ranges


def heading(names: Sequence[str], directory: str) -> str:
    """A short name for a family: its name pattern, the names' shared prefix
    and suffix (``*_beach_datasheet.csv``), or the directory and extension."""
    pattern = name_pattern(names)
    if pattern:
        return pattern[0]
    prefix = os.path.commonprefix(list(names))
    rests = [n[len(prefix) :] for n in names]
    suffix = os.path.commonprefix([r[::-1] for r in rests])[::-1]
    ext = (
        PurePosixPath(names[0]).suffix if len({PurePosixPath(n).suffix for n in names}) == 1 else ""
    )
    if len(prefix) + len(suffix) - len(ext) >= 3:
        return f"{prefix}*{suffix}"
    return f"{directory.rsplit('/', 1)[-1] or directory}/*{ext}"
