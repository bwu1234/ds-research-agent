"""The Ollama version check: a refusal list, recorded versions, no exact pin."""

from pathlib import Path

from ds_research_agent.ledger import Ledger
from eval.kramabench.run import resume_problems, server_version_problem
from tests.test_ledger_programs import _batch_and_run

REFUSED = ("0.40.0", "0.40.3")


def test_a_version_off_the_refusal_list_runs() -> None:
    assert server_version_problem("0.35.1", REFUSED) is None
    assert server_version_problem("0.40.4-rc0", REFUSED) is None


def test_a_listed_version_is_refused() -> None:
    assert "0.40.3" in (server_version_problem("0.40.3", REFUSED) or "")


def test_an_unreadable_server_version_is_refused() -> None:
    assert server_version_problem(None, ()) is not None


def test_resume_refuses_a_changed_server_version(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    _batch_and_run(ledger)
    old = ledger.batch("b").model_copy(update={"ollama_version": "0.35.1"})
    same = old.model_copy()
    newer = old.model_copy(update={"ollama_version": "0.40.4-rc0"})
    assert resume_problems(old, same) == []
    assert resume_problems(old, newer) == ["Ollama version (0.35.1 -> 0.40.4-rc0)"]


def test_resume_cannot_check_a_batch_recorded_without_a_version(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    _batch_and_run(ledger)
    old = ledger.batch("b").model_copy(update={"ollama_version": None})
    assert resume_problems(old, old.model_copy(update={"ollama_version": "0.40.4"})) == []
