"""Deterministic dataset-card profiler.

Walks the data root in sorted order and writes one Markdown card per file plus
a manifest. No LLM, no clock, no randomness: the same files and profiler
version produce byte-identical cards. Card text quotes data values, which are
untrusted; cards are evidence for the agent, never instructions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ds_research_agent.catalogue.models import (
    CatalogueEntry,
    Manifest,
    ParseStatus,
    SkippedPath,
)
from ds_research_agent.config import CatalogueSettings

# Bump whenever card text or entry fields change for the same input; a new
# version means a new catalogue generation and a full reindex.
PROFILER_VERSION = 1

# Reader the sandbox is expected to use, by format. Package set decided in D3.
READERS = {
    "csv": "pandas.read_csv",
    "json": "json / pandas.json_normalize",
}

# Tried in order; latin-1 decodes any byte sequence, so it is last.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
DELIMITERS = (",", ";", "\t", "|")
SNIFF_BYTES = 64 * 1024

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ProfileError(Exception):
    """The file has a supported format but could not be parsed."""


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    type: str
    nulls: int
    examples: tuple[str, ...]
    minimum: str | None = None
    maximum: str | None = None


@dataclass(frozen=True)
class TableProfile:
    title: str | None
    rows: int
    sampled_rows: int
    columns: tuple[ColumnProfile, ...]
    sample: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Profile:
    """What a format-specific profiler learned, before rendering."""

    facts: tuple[tuple[str, str], ...]
    tables: tuple[TableProfile, ...]


# --- value typing -----------------------------------------------------------


def _value_type(v: str) -> str:
    if _INT.match(v):
        return "integer"
    if _FLOAT.match(v):
        return "float"
    if v.lower() in ("true", "false"):
        return "boolean"
    if _DATE.match(v):
        return "date"
    return "string"


def _column_type(types: set[str]) -> str:
    if not types:
        return "empty"
    if len(types) == 1:
        return next(iter(types))
    if types == {"integer", "float"}:
        return "float"
    return "string"


def _profile_column(name: str, values: Sequence[str | None], s: CatalogueSettings) -> ColumnProfile:
    present = [v for v in values if v is not None and v != ""]
    ctype = _column_type({_value_type(v) for v in present})
    examples: list[str] = []
    for v in present:
        if v not in examples:
            examples.append(v)
        if len(examples) == s.max_example_values:
            break
    lo = hi = None
    if present and ctype in ("integer", "float"):
        nums = [float(v) for v in present]
        lo, hi = present[nums.index(min(nums))], present[nums.index(max(nums))]
    elif present and ctype == "date":
        lo, hi = min(present), max(present)
    return ColumnProfile(
        name=name,
        type=ctype,
        nulls=len(values) - len(present),
        examples=tuple(examples),
        minimum=lo,
        maximum=hi,
    )


def _table(
    title: str | None,
    header: Sequence[str],
    rows: Iterable[Sequence[str | None]],
    s: CatalogueSettings,
) -> TableProfile:
    sampled: list[Sequence[str | None]] = []
    total = 0
    for row in rows:
        total += 1
        if len(sampled) < s.sample_rows:
            sampled.append(row)
    columns = tuple(
        _profile_column(name, [r[i] if i < len(r) else None for r in sampled], s)
        for i, name in enumerate(header)
    )
    sample = tuple(tuple("" if v is None else v for v in r) for r in sampled[: s.card_sample_rows])
    return TableProfile(
        title=title, rows=total, sampled_rows=len(sampled), columns=columns, sample=sample
    )


# --- formats ----------------------------------------------------------------


def _decode(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc == "utf-8-sig":
            enc = "utf-8 with BOM" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        return text, enc
    raise AssertionError("latin-1 decodes every byte string")


def _delimiter(head: str) -> str:
    first = head.splitlines()[0] if head else ""
    counts = [(first.count(d), -i, d) for i, d in enumerate(DELIMITERS)]
    best = max(counts)
    return best[2] if best[0] > 0 else ","


def profile_csv(path: Path, s: CatalogueSettings) -> Profile:
    # Decodes the whole file to pick an encoding; acceptable for D0 fixtures.
    # D2 must stream large files (wildfire is about 1 GB).
    text, encoding = _decode(path)
    delim = _delimiter(text[:SNIFF_BYTES])
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    try:
        header = next(reader)
    except StopIteration:
        raise ProfileError("empty file") from None
    except csv.Error as e:
        raise ProfileError(f"csv: {e}") from e

    def rows() -> Iterator[list[str]]:
        try:
            yield from reader
        except csv.Error as e:
            raise ProfileError(f"csv line {reader.line_num}: {e}") from e

    table = _table(None, header, rows(), s)
    facts = (("Encoding", encoding), ("Delimiter", repr(delim)))
    return Profile(facts=facts, tables=(table,))


def _scalar(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float | str):
        return str(v)
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def _records_table(title: str | None, records: list[Any], s: CatalogueSettings) -> TableProfile:
    keys: list[str] = []
    for r in records[: s.sample_rows]:
        if isinstance(r, dict):
            keys.extend(k for k in r if k not in keys)
    rows = ([_scalar(r.get(k)) if isinstance(r, dict) else None for k in keys] for r in records)
    return _table(title, keys, rows, s)


def profile_json(path: Path, s: CatalogueSettings) -> Profile:
    size = path.stat().st_size
    if size > s.max_json_bytes:
        raise ProfileError(f"{size} bytes exceeds max_json_bytes {s.max_json_bytes}")
    text, encoding = _decode(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProfileError(f"json: {e}") from e

    facts: list[tuple[str, str]] = [("Encoding", encoding)]
    tables: list[TableProfile] = []
    if isinstance(data, list):
        facts.append(("Top level", f"array of {len(data)} items"))
        if any(isinstance(r, dict) for r in data):
            tables.append(_records_table(None, data, s))
    elif isinstance(data, dict):
        facts.append(("Top level", f"object with keys {', '.join(map(str, data))}"))
        for k, v in data.items():
            if isinstance(v, list) and any(isinstance(r, dict) for r in v):
                tables.append(_records_table(f"`{k}`", v, s))
            else:
                facts.append((f"`{k}`", _cell(_scalar(v) or "null", s)))
    else:
        facts.append(("Top level", type(data).__name__))
    return Profile(facts=tuple(facts), tables=tuple(tables))


PROFILERS = {"csv": profile_csv, "json": profile_json}


# --- rendering --------------------------------------------------------------


def _clip(v: str, s: CatalogueSettings) -> str:
    v = " ".join(v.split())
    return v if len(v) <= s.max_cell_chars else v[: s.max_cell_chars - 1] + "…"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace("|", "\\|")


def _cell(v: str, s: CatalogueSettings) -> str:
    return _escape(_clip(v, s))


def _render_table(t: TableProfile, s: CatalogueSettings) -> list[str]:
    out: list[str] = []
    if t.title:
        out += [f"## {t.title}", ""]
    stats = (
        f"Statistics from all {t.rows} rows."
        if t.sampled_rows == t.rows
        else f"Statistics from the first {t.sampled_rows} of {t.rows} rows."
    )
    out += [f"Rows: {t.rows}. Columns: {len(t.columns)}. {stats}", ""]
    out += ["| Column | Type | Nulls | Range | Example values |", "|---|---|---|---|---|"]
    for c in t.columns:
        rng = ""
        if c.minimum is not None and c.maximum is not None:
            rng = f"{_cell(c.minimum, s)} to {_cell(c.maximum, s)}"
        # Quote strings so a comma inside a value can't read as a separator.
        examples = ", ".join(
            _escape(json.dumps(_clip(e, s), ensure_ascii=False) if c.type == "string" else e)
            for e in c.examples
        )
        out.append(f"| {_cell(c.name, s)} | {c.type} | {c.nulls} | {rng} | {examples} |")
    if t.sample:
        out += ["", f"First {len(t.sample)} rows:", ""]
        out.append("| " + " | ".join(_cell(c.name, s) for c in t.columns) + " |")
        out.append("|" + "---|" * len(t.columns))
        for row in t.sample:
            cells = [_cell(row[i], s) if i < len(row) else "" for i in range(len(t.columns))]
            out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def render_card(
    front: dict[str, str],
    profile: Profile | None,
    status: ParseStatus,
    error: str | None,
    s: CatalogueSettings,
) -> str:
    # Every value is a string so rag-toolkit's equals/any_of filters match it.
    fm = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10_000)
    name = front["file_path"].rsplit("/", 1)[-1]
    lines = ["---", fm.rstrip("\n"), "---", f"# {name} ({front['domain']})", ""]
    lines += [f"Path: `{front['file_path']}`", f"Format: {front['format']}"]
    lines.append(f"Size: {front['size']} bytes")
    reader = READERS.get(front["format"])
    if reader:
        lines.append(f"Reader: {reader}")
    if status != "ok":
        lines.append(f"Parse status: {status}" + (f" ({error})" if error else ""))
    if profile is not None:
        lines += [f"{k}: {v}" for k, v in profile.facts]
    lines.append("")
    if profile is not None:
        for t in profile.tables:
            lines += _render_table(t, s)
    return "\n".join(lines).rstrip("\n") + "\n"


# --- catalogue build ----------------------------------------------------------


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def catalogue_id(domain: str, file_path: str) -> str:
    return f"{domain}/{hashlib.sha256(file_path.encode()).hexdigest()[:10]}"


def _walk(root: Path) -> tuple[list[Path], list[SkippedPath]]:
    files: list[Path] = []
    skipped: list[SkippedPath] = []
    for p in sorted(root.rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            if p.is_file():
                skipped.append(SkippedPath(path=rel, reason="hidden"))
            continue
        if p.is_symlink():
            skipped.append(SkippedPath(path=rel, reason="symlink"))
        elif p.is_file():
            if "/" not in rel:
                skipped.append(SkippedPath(path=rel, reason="outside any domain directory"))
            elif "\n" in rel or "\r" in rel:
                skipped.append(SkippedPath(path=rel, reason="newline in path"))
            else:
                files.append(p)
    return files, skipped


def profile_file(path: Path, root: Path, s: CatalogueSettings) -> tuple[CatalogueEntry, str]:
    rel = path.relative_to(root).as_posix()
    domain = rel.split("/", 1)[0]
    fmt = path.suffix.lower().lstrip(".") or "none"
    raw = path.read_bytes()
    cid = catalogue_id(domain, rel)

    profiler = PROFILERS.get(fmt)
    profile: Profile | None = None
    status: ParseStatus = "ok"
    error: str | None = None
    if profiler is None:
        status = "unsupported"
    else:
        try:
            profile = profiler(path, s)
        except ProfileError as e:
            status, error = "error", str(e)

    front = {
        "title": rel,
        "domain": domain,
        "catalogue_id": cid,
        "file_path": rel,
        "file_hash": _sha256(raw),
        "format": fmt,
        "size": str(len(raw)),
        "card_kind": "file",
        "parse_status": status,
        "profiler_version": str(PROFILER_VERSION),
    }
    card = render_card(front, profile, status, error, s)
    entry = CatalogueEntry(
        catalogue_id=cid,
        domain=domain,
        file_path=rel,
        size=len(raw),
        file_hash=front["file_hash"],
        format=fmt,
        document_id=f"{rel}.md",
        card_hash=_sha256(card.encode()),
        profiler_version=PROFILER_VERSION,
        parse_status=status,
        parse_error=error,
    )
    return entry, card


def build_catalogue(s: CatalogueSettings) -> Manifest:
    """Profile ``s.data_root`` into ``s.cards_dir`` and write the manifest.

    Cards are written to a staging directory and swapped in only after every
    file has been profiled, so a failed build leaves the previous cards intact.
    """
    root = s.data_root.resolve()
    files, skipped = _walk(root)

    staging = s.cards_dir.with_name(s.cards_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    entries: list[CatalogueEntry] = []
    seen: dict[str, str] = {}
    for path in files:
        entry, card = profile_file(path, root, s)
        if entry.catalogue_id in seen:
            raise RuntimeError(
                f"catalogue ID collision: {entry.file_path} and {seen[entry.catalogue_id]}"
            )
        seen[entry.catalogue_id] = entry.file_path
        out = staging / entry.document_id
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(card, encoding="utf-8")
        entries.append(entry)

    manifest = Manifest(
        profiler_version=PROFILER_VERSION, corpus=s.corpus, entries=entries, skipped=skipped
    )
    if s.cards_dir.exists():
        shutil.rmtree(s.cards_dir)
    staging.rename(s.cards_dir)
    s.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    s.manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest
