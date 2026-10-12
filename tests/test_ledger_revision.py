import sqlite3
from pathlib import Path

from ds_research_agent.ledger import Ledger, Run
from ds_research_agent.ledger.store import _SCHEMA
from tests.test_ledger_programs import _batch_and_run


def _run(ledger: Ledger, task: str, started: str, revision: str | None = None) -> None:
    ledger.add_run(
        Run(
            run_id=f"b/{task}/0", batch_id="b", task_key=task, parent_task_key=task, split="dev",
            condition="given_files", repeat=0, input_manifest={}, access_policy="a",
            catalogue_generation=None, started=started, repo_revision=revision,
        )
    )  # fmt: skip


def test_schema_3_ledger_gains_the_column_and_keeps_old_rows(tmp_path: Path) -> None:
    path = tmp_path / "v3.sqlite3"
    old = _SCHEMA.replace(
        "    failure_category TEXT,\n    repo_revision TEXT,\n", "    failure_category TEXT,\n"
    )
    assert old != _SCHEMA
    db = sqlite3.connect(path)
    db.executescript(old)
    db.execute("INSERT INTO meta VALUES ('schema_version', '3')")
    db.commit()
    db.close()
    ledger = Ledger(path)
    rid = _batch_and_run(ledger)  # new code can write a run into the upgraded table
    assert ledger.runs("b")[0].repo_revision is None and ledger.runs("b")[0].run_id == rid
    ledger.close()
    Ledger(path).close()  # reopening is a no-op


def test_run_revision_round_trips(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    _batch_and_run(ledger)
    _run(ledger, "k2", "2026-10-12T00:00:00", "abc1234")
    assert {r.task_key: r.repo_revision for r in ledger.runs("b")} == {"k": None, "k2": "abc1234"}


def test_backfill_fills_only_missing_revisions_inside_the_window(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    _batch_and_run(ledger)
    ledger.set_run_revision("b", "seed")  # the helper's run starts at the real clock time
    _run(ledger, "early", "2026-10-11T22:00:00")
    _run(ledger, "late", "2026-10-12T03:00:00")
    _run(ledger, "known", "2026-10-11T23:00:00", "keep")
    assert ledger.set_run_revision("b", "old", started_before="2026-10-12T02:00:00") == 1
    got = {r.task_key: r.repo_revision for r in ledger.runs("b")}
    assert got == {"k": "seed", "early": "old", "late": None, "known": "keep"}
    assert ledger.set_run_revision("b", "new", started_from="2026-10-12T02:00:00") == 1
    assert {r.task_key: r.repo_revision for r in ledger.runs("b")}["late"] == "new"
    assert ledger.set_run_revision("b", "other") == 0  # nothing left to fill
