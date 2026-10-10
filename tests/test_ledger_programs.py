"""Schema-2 ledger tables (agent runs) and the in-place upgrade from schema 1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ds_research_agent.ledger import (
    SCHEMA_VERSION,
    Batch,
    Ledger,
    LedgerError,
    Program,
    Run,
    SubmissionRow,
    VerificationRow,
    utc_now,
)
from ds_research_agent.ledger.store import _SCHEMA


def _batch_and_run(ledger: Ledger) -> str:
    ledger.add_batch(
        Batch(
            batch_id="b", created=utc_now(), condition="given_files", task_set="t",
            task_keys=("k",), repeats=1, config={}, config_sha256="c", repo_revision=None,
            repo_dirty=None, kramabench_commit="k", split_sha256="s", model_name="m",
            model_settings={}, ollama_version=None, answer_type_visible=False,
            scoring_profile="p", system_prompt_sha256="h",
        )
    )  # fmt: skip
    ledger.add_run(
        Run(
            run_id="b/k/0", batch_id="b", task_key="k", parent_task_key="k", split="dev",
            condition="given_files", repeat=0, input_manifest={}, access_policy="a",
            catalogue_generation=None, started=utc_now(),
        )
    )  # fmt: skip
    return "b/k/0"


def test_programs_submission_verification_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    rid = _batch_and_run(ledger)
    p = Program(
        run_id=rid, seq=0, step=1, call_index=0, kind="cell", session="s", code="x = 1",
        status="ok", wall_s=0.1, audit_complete=True, observed_reads=["/data/a.csv"],
        record={"cell": 1, "stdout": ""},
    )  # fmt: skip
    ledger.add_program(p)
    sub = SubmissionRow(
        run_id=rid, step=2, value=None, files_used=["/data/a.csv"], program="print()",
        assumptions=[],
    )  # fmt: skip
    ledger.put_submission(sub)
    v = VerificationRow(
        run_id=rid, comparator="reproduction-v1", rerun_exit_code=0, rerun_answered=True,
        rerun_answer=None, reproduced=True, audit_complete=True, observed_files=["/data/a.csv"],
        claimed_files=["/data/a.csv"], access_verified=True, detail=[],
    )  # fmt: skip
    ledger.put_verification(v)
    assert ledger.programs(rid) == [p]
    assert ledger.submission(rid) == sub  # a JSON null answer stays a value
    assert ledger.verification(rid) == v
    # An unfinished run is discarded with its agent records.
    assert ledger.discard_unfinished("b") == ["k"]
    assert ledger.programs(rid) == [] and ledger.submission(rid) is None


def test_schema_1_ledger_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(_SCHEMA)
    db.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    db.commit()
    db.close()
    ledger = Ledger(path)
    rid = _batch_and_run(ledger)
    assert ledger.programs(rid) == []
    ledger.close()
    version = sqlite3.connect(path).execute("SELECT value FROM meta").fetchone()[0]
    assert int(version) == SCHEMA_VERSION == 3


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "new.sqlite3"
    Ledger(path).close()
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value = '9'")
    db.commit()
    db.close()
    with pytest.raises(LedgerError):
        Ledger(path)
