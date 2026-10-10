"""SQLite run ledger.

One database holds evaluation batches (one condition and configuration over a
task set), runs (one task attempt), steps (one model request each), answers,
and scores, and for agent runs (schema 2, D3) programs (exploration cells and
the final program's fresh rerun), the submission, and its verification, and
(schema 3) one failure-taxonomy label per failed run. It
quotes prompts, model output, data, and program output, so it lives under the
ignored ``data/`` tree and is never committed. Older ledgers are upgraded in
place by adding the newer tables.

Timestamps are timezone-aware UTC ISO strings. JSON columns hold canonical
JSON (sorted keys) so equal values compare equal as text.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    condition TEXT NOT NULL,
    task_set TEXT NOT NULL,
    task_keys_json TEXT NOT NULL,
    repeats INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    repo_revision TEXT,
    repo_dirty INTEGER,
    kramabench_commit TEXT NOT NULL,
    split_sha256 TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_settings_json TEXT NOT NULL,
    ollama_version TEXT,
    answer_type_visible INTEGER NOT NULL,
    scoring_profile TEXT NOT NULL,
    system_prompt_sha256 TEXT NOT NULL,
    replay_of TEXT REFERENCES batches(batch_id),
    note TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    task_key TEXT NOT NULL,
    parent_task_key TEXT NOT NULL,
    split TEXT NOT NULL,
    condition TEXT NOT NULL,
    repeat INTEGER NOT NULL,
    input_manifest_json TEXT NOT NULL,
    access_policy TEXT NOT NULL,
    catalogue_generation TEXT,
    started TEXT NOT NULL,
    ended TEXT,
    wall_s REAL,
    stop_reason TEXT,
    error TEXT,
    failure_category TEXT,
    UNIQUE (batch_id, task_key, repeat)
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    prompt_eval_tokens INTEGER,
    prompt_eval_s REAL,
    output_tokens INTEGER,
    eval_s REAL,
    load_s REAL,
    thinking_chars INTEGER,
    tool_call_repairs INTEGER NOT NULL DEFAULT 0,
    wall_s REAL,
    done_reason TEXT,
    error TEXT,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS answers (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    answered INTEGER NOT NULL,
    value_json TEXT,
    parse_status TEXT NOT NULL,
    abstained INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    profile TEXT NOT NULL,
    answer_type TEXT NOT NULL,
    metric TEXT NOT NULL,
    score REAL NOT NULL,
    strict INTEGER NOT NULL,
    -- NULL for conditions without a program (no-tools, inlined files).
    verified_success INTEGER,
    scored TEXT NOT NULL,
    PRIMARY KEY (run_id, profile)
);
"""

# Schema 2 (D3): agent runs. Additive, so schema-1 ledgers upgrade in place.
_SCHEMA_2 = """
CREATE TABLE IF NOT EXISTS programs (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    seq INTEGER NOT NULL,
    step INTEGER NOT NULL,
    call_index INTEGER NOT NULL,
    kind TEXT NOT NULL,  -- cell | final_rerun
    session TEXT,
    code TEXT NOT NULL,
    status TEXT NOT NULL,  -- cell status, or the rerun's exit code as text
    wall_s REAL,
    audit_complete INTEGER NOT NULL,
    observed_reads_json TEXT NOT NULL,
    replan_requested INTEGER NOT NULL DEFAULT 0,
    -- The full CellResult or SandboxRun, for replay and failure analysis.
    record_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS submissions (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    step INTEGER NOT NULL,
    value_json TEXT,
    files_used_json TEXT NOT NULL,
    program TEXT NOT NULL,
    assumptions_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    comparator TEXT NOT NULL,
    rerun_exit_code INTEGER,
    rerun_answered INTEGER NOT NULL,
    rerun_answer_json TEXT,
    reproduced INTEGER NOT NULL,
    audit_complete INTEGER NOT NULL,
    observed_files_json TEXT NOT NULL,
    claimed_files_json TEXT NOT NULL,
    access_verified INTEGER NOT NULL,
    detail_json TEXT NOT NULL
);
"""

# Schema 3 (D3): failure-taxonomy labels, one per failed run. Additive.
_SCHEMA_3 = """
CREATE TABLE IF NOT EXISTS failure_labels (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    taxonomy TEXT NOT NULL,  -- taxonomy version
    category TEXT NOT NULL,
    source TEXT NOT NULL,  -- rule | manual
    note TEXT NOT NULL,
    labelled TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=True)


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Batch(_Row):
    batch_id: str
    created: str
    condition: str
    task_set: str
    task_keys: tuple[str, ...]
    repeats: int
    config: dict[str, Any]
    config_sha256: str
    repo_revision: str | None
    repo_dirty: bool | None
    kramabench_commit: str
    split_sha256: str
    model_name: str
    model_settings: dict[str, Any]
    ollama_version: str | None
    answer_type_visible: bool
    scoring_profile: str
    system_prompt_sha256: str
    replay_of: str | None = None
    note: str | None = None


class Run(_Row):
    run_id: str
    batch_id: str
    task_key: str
    parent_task_key: str
    split: str
    condition: str
    repeat: int
    input_manifest: dict[str, Any]
    access_policy: str
    catalogue_generation: str | None
    started: str
    ended: str | None = None
    wall_s: float | None = None
    stop_reason: str | None = None
    error: str | None = None
    failure_category: str | None = None


class Step(_Row):
    run_id: str
    seq: int
    kind: str  # "model"
    request_sha256: str
    request: dict[str, Any]
    response: dict[str, Any] | None
    prompt_eval_tokens: int | None = None
    prompt_eval_s: float | None = None
    output_tokens: int | None = None
    eval_s: float | None = None
    load_s: float | None = None
    thinking_chars: int | None = None
    tool_call_repairs: int = 0
    wall_s: float | None = None
    done_reason: str | None = None
    error: str | None = None


class Answer(_Row):
    run_id: str
    answered: bool
    value: Any = None
    parse_status: str
    abstained: bool = False


class Program(_Row):
    run_id: str
    seq: int
    step: int
    call_index: int
    kind: str
    session: str | None
    code: str
    status: str
    wall_s: float | None
    audit_complete: bool
    observed_reads: list[str]
    replan_requested: bool = False
    record: dict[str, Any]


class SubmissionRow(_Row):
    run_id: str
    step: int
    value: Any
    files_used: list[str]
    program: str
    assumptions: list[str]


class VerificationRow(_Row):
    run_id: str
    comparator: str
    rerun_exit_code: int | None
    rerun_answered: bool
    rerun_answer: Any
    reproduced: bool
    audit_complete: bool
    observed_files: list[str]
    claimed_files: list[str]
    access_verified: bool
    detail: list[str]


class FailureLabel(_Row):
    run_id: str
    taxonomy: str
    category: str
    source: str
    note: str
    labelled: str


class Score(_Row):
    run_id: str
    profile: str
    answer_type: str
    metric: str
    score: float
    strict: bool
    verified_success: bool | None
    scored: str


_JSON_COLS = {
    "task_keys": "task_keys_json",
    "config": "config_json",
    "model_settings": "model_settings_json",
    "input_manifest": "input_manifest_json",
    "request": "request_json",
    "response": "response_json",
    "value": "value_json",
    "observed_reads": "observed_reads_json",
    "record": "record_json",
    "files_used": "files_used_json",
    "assumptions": "assumptions_json",
    "rerun_answer": "rerun_answer_json",
    "observed_files": "observed_files_json",
    "claimed_files": "claimed_files_json",
    "detail": "detail_json",
}
# JSON columns where a JSON null is a value, not a missing one.
_NULLABLE_VALUES = {"value", "rerun_answer"}


def _to_row(rec: _Row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.model_dump(mode="json").items():
        if k in _JSON_COLS:
            out[_JSON_COLS[k]] = None if v is None and k not in _NULLABLE_VALUES else canonical(v)
        elif isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _from_row[R: _Row](cls: type[R], row: sqlite3.Row) -> R:
    data: dict[str, Any] = {}
    back = {v: k for k, v in _JSON_COLS.items()}
    for k in row.keys():
        v = row[k]
        if k in back:
            data[back[k]] = None if v is None else json.loads(v)
        else:
            data[k] = v
    return cls.model_validate(data)


class LedgerError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)
        row = self._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is not None and not 1 <= int(row[0]) <= SCHEMA_VERSION:
            raise LedgerError(f"{path}: schema {row[0]}, expected {SCHEMA_VERSION}")
        self._db.executescript(_SCHEMA_2)
        self._db.executescript(_SCHEMA_3)
        self._db.execute(
            "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def transaction(self) -> Generator[None]:
        self._db.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def _insert(self, table: str, rec: _Row, *, replace: bool = False) -> None:
        row = _to_row(rec)
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self._db.execute(f"{verb} INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))

    def add_batch(self, b: Batch) -> None:
        self._insert("batches", b)

    def add_run(self, r: Run) -> None:
        self._insert("runs", r)

    def finish_run(
        self, run_id: str, *, ended: str, wall_s: float, stop_reason: str, error: str | None
    ) -> None:
        self._db.execute(
            "UPDATE runs SET ended = ?, wall_s = ?, stop_reason = ?, error = ? WHERE run_id = ?",
            (ended, wall_s, stop_reason, error, run_id),
        )

    def discard_unfinished(self, batch_id: str) -> list[str]:
        """Delete runs of ``batch_id`` that never finished (an interrupted
        task) with their steps, answer, and scores; return their task keys."""
        rows = self._db.execute(
            "SELECT run_id, task_key FROM runs WHERE batch_id = ? AND ended IS NULL",
            (batch_id,),
        ).fetchall()
        with self.transaction():
            for run_id, _ in rows:
                for table in (
                    "failure_labels", "verifications", "submissions", "programs", "scores",
                    "answers", "steps", "runs",
                ):  # fmt: skip
                    self._db.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        return [key for _, key in rows]

    def add_step(self, s: Step) -> None:
        self._insert("steps", s)

    def put_answer(self, a: Answer) -> None:
        self._insert("answers", a, replace=True)

    def put_score(self, s: Score) -> None:
        self._insert("scores", s, replace=True)

    def batch(self, batch_id: str) -> Batch:
        row = self._db.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if row is None:
            raise LedgerError(f"no batch {batch_id!r}")
        return _from_row(Batch, row)

    def batches(self) -> list[Batch]:
        rows = self._db.execute("SELECT * FROM batches ORDER BY created").fetchall()
        return [_from_row(Batch, r) for r in rows]

    def runs(self, batch_id: str) -> list[Run]:
        rows = self._db.execute(
            "SELECT * FROM runs WHERE batch_id = ? ORDER BY rowid", (batch_id,)
        ).fetchall()
        return [_from_row(Run, r) for r in rows]

    def steps(self, run_id: str) -> list[Step]:
        rows = self._db.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [_from_row(Step, r) for r in rows]

    def answer(self, run_id: str) -> Answer | None:
        row = self._db.execute("SELECT * FROM answers WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _from_row(Answer, row)

    def score(self, run_id: str, profile: str) -> Score | None:
        row = self._db.execute(
            "SELECT * FROM scores WHERE run_id = ? AND profile = ?", (run_id, profile)
        ).fetchone()
        return None if row is None else _from_row(Score, row)

    def add_program(self, p: Program) -> None:
        self._insert("programs", p)

    def put_submission(self, sub: SubmissionRow) -> None:
        self._insert("submissions", sub, replace=True)

    def put_verification(self, v: VerificationRow) -> None:
        self._insert("verifications", v, replace=True)

    def programs(self, run_id: str) -> list[Program]:
        rows = self._db.execute(
            "SELECT * FROM programs WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [_from_row(Program, r) for r in rows]

    def submission(self, run_id: str) -> SubmissionRow | None:
        row = self._db.execute("SELECT * FROM submissions WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _from_row(SubmissionRow, row)

    def verification(self, run_id: str) -> VerificationRow | None:
        row = self._db.execute("SELECT * FROM verifications WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _from_row(VerificationRow, row)

    def put_failure_label(self, label: FailureLabel) -> None:
        self._insert("failure_labels", label, replace=True)

    def failure_label(self, run_id: str) -> FailureLabel | None:
        row = self._db.execute(
            "SELECT * FROM failure_labels WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else _from_row(FailureLabel, row)
