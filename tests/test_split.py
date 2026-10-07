from pathlib import Path

import pytest

from eval.kramabench.split import SplitError, frozen_split, split_for
from eval.kramabench.tasks import load_tasks, resolve_sources
from tests.kramabench_synthetic import make_layout, settings_for


def test_tiny_tasks_are_keyed_by_workload(tmp_path: Path) -> None:
    ev, _ = make_layout(tmp_path)
    tasks = load_tasks(ev)
    assert tasks["legal-tiny/legal-hard-2"].parent_key == "legal/legal-hard-2"
    assert tasks["legal-tiny/legal-hard-2"].answer_type != tasks["legal/legal-hard-2"].answer_type
    assert len(tasks) == 6 * 4 + 2


def test_split_is_stratified_deterministic_and_keeps_smoke_parents_in_dev(
    tmp_path: Path,
) -> None:
    s = settings_for(tmp_path)
    a, tasks = split_for(s)
    b, _ = split_for(s)
    assert a.canonical() == b.canonical()
    assert set(a.dev).isdisjoint(a.holdout)
    assert len(a.dev) + len(a.holdout) == 24
    for d in {t.domain for t in tasks.values()}:
        dev = [k for k in a.dev if tasks[k].domain == d]
        assert len(dev) == 2  # round_half_up(0.5 * 4)
        assert {tasks[k].difficulty for k in dev} == {"easy", "hard"}
    assert {"legal/legal-hard-2", "environment/environment-easy-1"} <= set(a.dev)
    assert set(a.sample) <= set(a.dev)
    assert len(a.sample) == 6


def test_sample_skips_tasks_with_unresolved_sources(tmp_path: Path) -> None:
    s = settings_for(tmp_path, sample_per_domain=2)
    split, tasks = split_for(s)
    for k in split.sample:
        t = tasks[k]
        res = resolve_sources(t.domain, t.data_sources, s.kramabench.visible_root)
        assert all(r.tier != "unresolved" for r in res)


def test_seed_changes_split_and_frozen_hash_is_enforced(tmp_path: Path) -> None:
    s = settings_for(tmp_path)
    split, _ = split_for(s)
    with pytest.raises(SplitError, match="not set"):
        frozen_split(s)
    frozen = s.model_copy(
        update={"eval": s.eval.model_copy(update={"split_sha256": split.sha256()})}
    )
    assert frozen_split(frozen)[0] == split
    reseeded = frozen.model_copy(
        update={"eval": frozen.eval.model_copy(update={"split_seed": s.eval.split_seed + 1})}
    )
    assert split_for(reseeded)[0].sha256() != split.sha256()
    with pytest.raises(SplitError, match="differs"):
        frozen_split(reseeded)


def test_resolution_tiers(tmp_path: Path) -> None:
    base = tmp_path / "legal" / "input"
    (base / "sub" / "CSVs").mkdir(parents=True)
    (base / "top.csv").write_text("a\n")
    (base / "sub" / "CSVs" / "deep.csv").write_text("a\n")
    (base / "Mixed Data").mkdir()
    (base / "Mixed Data" / "m.csv").write_text("a\n")
    res = resolve_sources(
        "legal", ("top.csv", "deep.csv", "Mixed data/", "typo.csv", "*.csv"), tmp_path
    )
    assert [r.tier for r in res] == ["exact", "nested", "casefold", "unresolved", "exact"]
    assert res[2].files == ("legal/input/Mixed Data/m.csv",)
