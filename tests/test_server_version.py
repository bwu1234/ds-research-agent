"""The Ollama version check: a refusal list, recorded versions, no exact pin."""

from pathlib import Path

import httpx
import pytest

from ds_research_agent.config import ModelSettings, load_settings
from ds_research_agent.ledger import Ledger
from eval.kramabench import run
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


def test_resume_refuses_a_batch_recorded_without_a_version(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    _batch_and_run(ledger)
    old = ledger.batch("b").model_copy(update={"ollama_version": None})
    new = old.model_copy(update={"ollama_version": "0.40.4"})
    assert any("not recorded" in p for p in resume_problems(old, new))


def test_resume_ignores_the_legacy_pin_and_the_refusal_list(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite3")
    _batch_and_run(ledger)
    base = ledger.batch("b").model_copy(update={"ollama_version": "0.35.1"})
    legacy = base.model_copy(
        update={"model_settings": {"think": False, "server_version": "0.35.1"}}
    )
    current = base.model_copy(
        update={"model_settings": {"think": False, "refuse_server_versions": ["0.40.3"]}}
    )
    assert resume_problems(legacy, current) == []
    changed = base.model_copy(update={"model_settings": {"think": "low"}})
    assert resume_problems(legacy, changed) == ["model_settings"]


def test_legacy_stored_model_settings_still_validate_for_replay() -> None:
    stored = {
        "host": "http://127.0.0.1:11434", "keep_alive": "30m", "name": "m",
        "options": {"seed": 0}, "request_timeout_s": 900.0, "server_version": "0.35.1",
        "think": False,
    }  # fmt: skip
    model = ModelSettings.model_validate(stored)
    assert model.refuse_server_versions == () and not hasattr(model, "server_version")


def test_a_settings_file_with_the_legacy_pin_loads_with_a_warning(tmp_path: Path) -> None:
    cfg = Path("config/example.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace(
        "  refuse_server_versions:", "  server_version: 0.35.1\n  refuse_server_versions:"
    )
    path = tmp_path / "old.yaml"
    path.write_text(cfg, encoding="utf-8")
    with pytest.warns(UserWarning, match="server_version is no longer used"):
        settings = load_settings(path)
    assert settings.model.refuse_server_versions


@pytest.mark.parametrize("body", [{"version": None}, {"version": ""}, {"version": 7}, [], {}])
def test_an_unreadable_version_field_is_not_a_version(
    body: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run.httpx, "get", lambda *a, **k: httpx.Response(200, json=body))
    assert run.ollama_version("http://x") is None


def test_a_readable_version_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run.httpx, "get", lambda *a, **k: httpx.Response(200, json={"version": "0.40.4-rc0"})
    )
    assert run.ollama_version("http://x") == "0.40.4-rc0"
