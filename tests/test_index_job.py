import subprocess
from pathlib import Path

import pytest

from ds_research_agent.catalogue.index_job import (
    CARRY_METADATA,
    _rag_cli_logged,
    check_checkout,
    rag_config,
)
from ds_research_agent.config import CatalogueSettings, RagServiceSettings

BASE = {
    "paths": {"corpus_dir": "data/corpora/baseline", "index_dir": "data/index"},
    "corpora": {
        "active": ["baseline"],
        "registry": {"baseline": {"documents_dir": "x"}, "edgar": {"documents_dir": "y"}},
    },
    "chunking": {"chunk_size": 1000, "carry_metadata": ["company"], "header": {"template": "{x}"}},
    "retrieval": {"top_k": 20, "expansion": {"provider": "hyde"}, "web_search": {"enabled": True}},
}


def settings(tmp_path: Path, checkout: Path, revision: str = "0" * 40) -> RagServiceSettings:
    return RagServiceSettings(
        checkout=checkout,
        revision=revision,
        python=Path("/usr/bin/python3"),
        base_config=tmp_path / "base.yaml",
        generated_config=tmp_path / "rag.yaml",
        index_dir=tmp_path / "index",
        index_timeout_s=10,
    )


def cat(tmp_path: Path) -> CatalogueSettings:
    return CatalogueSettings(
        data_root=tmp_path / "lake",
        cards_dir=tmp_path / "cards",
        manifest_path=tmp_path / "manifest.json",
        corpus="fixture-cards",
        sample_rows=10,
        card_sample_rows=1,
        max_example_values=1,
        max_cell_chars=10,
        max_json_bytes=10,
        text_preview_lines=2,
    )


def test_flattened_config_serves_only_card_corpus(tmp_path: Path) -> None:
    c = rag_config(BASE, cat(tmp_path), settings(tmp_path, tmp_path))
    assert "base" not in c
    assert list(c["corpora"]["registry"]) == ["fixture-cards"]
    assert c["corpora"]["active"] == ["fixture-cards"]
    assert c["chunking"]["carry_metadata"] == CARRY_METADATA
    assert c["chunking"]["chunk_size"] == 1000  # untouched base settings survive
    assert c["retrieval"]["expansion"]["provider"] == "none"
    assert c["retrieval"]["web_search"]["enabled"] is False
    assert c["retrieval"]["top_k"] == 20
    assert Path(c["paths"]["index_dir"]).is_absolute()


def test_base_with_its_own_base_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flatten"):
        rag_config({"base": "x.yaml"}, cat(tmp_path), settings(tmp_path, tmp_path))


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    (r / "f.txt").write_text("a")
    git(r, "add", ".")
    git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return r


def test_checkout_guard(tmp_path: Path, repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD")
    check_checkout(settings(tmp_path, repo, head))
    with pytest.raises(RuntimeError, match="pinned"):
        check_checkout(settings(tmp_path, repo))
    (repo / "f.txt").write_text("b")
    with pytest.raises(RuntimeError, match="local modifications"):
        check_checkout(settings(tmp_path, repo, head))


def fake_python(tmp_path: Path, body: str) -> RagServiceSettings:
    """Service settings whose interpreter is a shell script standing in for rag.cli."""
    exe = tmp_path / "python"
    exe.write_text(f"#!/bin/sh\necho started\n{body}\n")
    exe.chmod(0o755)
    s = settings(tmp_path, tmp_path)
    # 3 s, not 1: macOS may scan a new executable on its first run.
    return s.model_copy(update={"python": exe, "index_timeout_s": 3})


def test_logged_cli_keeps_output_when_it_times_out(tmp_path: Path) -> None:
    log = tmp_path / "index.log"
    with pytest.raises(RuntimeError, match=r"timed out after 3 s; see .*index\.log"):
        _rag_cli_logged(fake_python(tmp_path, "sleep 30"), log, "index", "--corpus", "c")
    text = log.read_text()
    assert "rag.cli --config" in text and "index --corpus c" in text
    assert text.rstrip().endswith("started")


def test_logged_cli_reports_a_failing_exit(tmp_path: Path) -> None:
    log = tmp_path / "index.log"
    with pytest.raises(RuntimeError, match="exited 3"):
        _rag_cli_logged(fake_python(tmp_path, "echo boom >&2; exit 3"), log, "index")
    assert log.read_text().splitlines()[-2:] == ["started", "boom"]
