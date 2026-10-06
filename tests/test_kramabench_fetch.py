import json
import subprocess
from pathlib import Path

import pytest

from ds_research_agent.config import KramaBenchSettings
from eval.kramabench.fetch import FetchError, check_layout, fetch, record_path, verify_store


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str]:
    """A synthetic stand-in for the benchmark repository."""
    repo = tmp_path / "upstream"
    files = {
        "data/legal/a.csv": "x,y\n1,2\n",
        "data/astronomy/sub/b.csv": "t\n3\n",
        "workload/legal.json": '[{"id": "legal-1", "answer": 42}]',
        "solutions/legal-1.py": "print(42)\n",
        "dr-input/legal-1/a.csv": "x,y\n1,2\n",
        "README.md": "readme\n",
    }
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git("init", "-q", cwd=repo)
    git("add", ".", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=repo)
    return repo, git("rev-parse", "HEAD", cwd=repo)


def settings(tmp_path: Path, repo: Path, commit: str) -> KramaBenchSettings:
    root = tmp_path / "kb"
    return KramaBenchSettings(
        repo_url=repo.as_uri(),
        commit=commit,
        checkout=root / "repo",
        visible_root=root / "visible",
        evaluator_root=root / "evaluator",
    )


def test_split_by_visibility(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    s = settings(tmp_path, *upstream)
    rec = fetch(s)

    visible = sorted(str(p.relative_to(s.visible_root)) for p in s.visible_root.rglob("*.csv"))
    assert visible == ["astronomy/sub/b.csv", "legal/a.csv"]
    # Nothing answer-bearing or duplicated reaches the agent-visible store.
    assert not any("workload" in str(p) or "solutions" in str(p) for p in s.visible_root.rglob("*"))
    assert (s.evaluator_root / "workload/legal.json").exists()
    assert (s.evaluator_root / "solutions/legal-1.py").exists()
    assert not (s.evaluator_root / "dr-input").exists()
    assert not (s.evaluator_root / "data").exists()

    assert rec["commit"] == upstream[1]
    assert rec["visible"]["files"] == 2
    assert rec["visible_by_domain"] == {"astronomy": 1, "legal": 1}
    assert rec["excluded_files"] == 1
    assert json.loads(record_path(s).read_text())["tree"] == rec["tree"]
    assert verify_store(s.visible_root) == []
    assert verify_store(s.evaluator_root) == []


def test_refetch_is_identical(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    s = settings(tmp_path, *upstream)
    fetch(s)
    first = (s.visible_root / "SHA256SUMS").read_bytes()
    fetch(s)
    assert (s.visible_root / "SHA256SUMS").read_bytes() == first


def test_verify_detects_drift(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    s = settings(tmp_path, *upstream)
    fetch(s)
    target = s.visible_root / "legal/a.csv"
    target.chmod(0o644)
    target.write_text("tampered\n")
    (s.visible_root / "legal/new.csv").write_text("z\n")
    assert verify_store(s.visible_root) == ["extra file legal/new.csv", "changed legal/a.csv"]


def test_checkout_at_wrong_commit_is_refused(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, commit = upstream
    s = settings(tmp_path, repo, commit)
    fetch(s)
    (s.checkout / "data/legal/a.csv").write_text("edited\n")
    with pytest.raises(FetchError, match="local changes"):
        fetch(s)


def test_overlapping_stores_are_refused(tmp_path: Path) -> None:
    s = KramaBenchSettings(
        repo_url="x",
        commit="0" * 40,
        checkout=tmp_path / "repo",
        visible_root=tmp_path / "repo" / "data",
        evaluator_root=tmp_path / "eval",
    )
    with pytest.raises(FetchError, match="overlap"):
        check_layout(s)
