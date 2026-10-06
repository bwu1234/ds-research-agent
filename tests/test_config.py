from pathlib import Path

import pytest
from pydantic import ValidationError

from ds_research_agent.config import load_settings

EXAMPLE = Path(__file__).parents[1] / "config" / "example.yaml"


def test_example_config_validates() -> None:
    s = load_settings(EXAMPLE)
    assert s.model.name == "qwen3.8:27b-mlx"
    assert s.mcp.protocol_version == "2026-07-28"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text().replace("keep_alive:", "keep_alivee:")
    p = tmp_path / "c.yaml"
    p.write_text(text)
    with pytest.raises(ValidationError, match="keep_alivee"):
        load_settings(p)


def test_invalid_think_level_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text().replace("think: medium", "think: high")
    p = tmp_path / "c.yaml"
    p.write_text(text)
    with pytest.raises(ValidationError, match="think"):
        load_settings(p)


def test_env_overrides_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSRA_MODEL__THINK", "low")
    assert load_settings(EXAMPLE).model.think == "low"
