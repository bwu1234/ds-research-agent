"""Local deterministic scorer: edge cases, and agreement with upstream.

The differential tests load the pinned upstream ``benchmark/metrics.py``
from the local evaluator store (no network) with its LLM and NLP imports
stubbed, and skip when the store is absent. They print nothing
benchmark-derived.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.config import load_settings
from eval.kramabench import scoring
from eval.kramabench.scoring import f1, f1_approximate, rae_score, score_answer, success
from eval.kramabench.tasks import Gold, load_tasks

LOCAL = Path(os.environ.get("DSRA_CONFIG", Path(__file__).parents[1] / "config" / "local.yaml"))


# --- edge cases -------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred", "target", "want"),
    [
        (42, 42, 1),
        ("42", 42, 1),
        (" 42 ", 42, 1),
        (42.0, 42, 1),
        (41, 42, 0),
        (3.14159, 3.1416, 1),  # rounding: |diff| < 1e-4 * magnitude
        (3.14, 3.1416, 0),
        ("12%", 12.0, 1),  # try_convert strips '%' without dividing
        (0.0000001, 0.0, 1),  # zero target: absolute 1e-6
        (0.001, 0.0, 0),
        ("Utah", "utah ", 1),
        ("Utah state", "Utah", 0),
        (5, "five", 0),
        (None, 42, 0),
        (True, 1, 1),  # bool is an int upstream
        ([1], 1, 0),
    ],
)
def test_success(pred: Any, target: Any, want: int) -> None:
    assert success(pred, target) == want


def test_rae_score() -> None:
    assert rae_score(10, 10) == 1.0
    assert rae_score(15, 10) == pytest.approx(1 / 1.5)
    assert rae_score("50%", 0.5) == 1.0  # str_to_float divides percentages
    assert rae_score("x", 10) == 0.0
    assert rae_score(1, 0) == 0.0  # zero target raises upstream; scores 0
    assert rae_score(float("nan"), 10) == 0.0  # deviation 4: upstream gives NaN
    assert rae_score(float("inf"), 10) == 0.0


def test_f1_lists_and_parsing() -> None:
    assert f1(["A", " b"], ["a", "b"]) == 1.0
    assert f1(["a"], ["a", "b"]) == pytest.approx(2 / 3)
    assert f1('["a", "b"]', ["a", "b"]) == 1.0
    assert f1("['a', 'b']", ["a", "b"]) == 1.0  # literal_eval where upstream used eval
    assert f1("a, b", ["a", "b"]) == 1.0  # comma fallback
    assert f1(["1.0000001", 2], [1, 2]) == 1.0
    assert f1([], []) == 1.0
    assert f1(5, ["a"]) == 0.0
    assert f1("x", "X ") == 1.0


def test_f1_never_executes_model_output(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    assert f1(payload, ["a"]) == 0.0
    assert not marker.exists()
    assert f1_approximate(payload, ["a"]) == 0.0
    assert not marker.exists()


def test_f1_approximate_is_deterministic() -> None:
    assert f1_approximate([1.005, 2], [1.0, 2.0]) == 1.0  # within 1%
    assert f1_approximate([1.02, 2], [1.0, 2.0]) == 0.5  # 2% misses; P = R = 1/2
    assert f1_approximate(["Fire Season"], ["fire season"]) == 1.0
    assert f1_approximate(["the fire season"], ["fire season"]) == 0.0  # no paraphrase judge
    assert f1_approximate(3, [3]) == 0.0


def test_strict_accuracy_and_unanswered() -> None:
    g = Gold(answer=100.0, answer_type="numeric_approximate")
    assert score_answer(100.9, g).strict
    assert not score_answer(101.1, g).strict
    assert score_answer(101.1, g).score == pytest.approx(1 / 1.011)
    assert not score_answer(True, g).strict
    exact = Gold(answer=["a", "b"], answer_type="list_exact")
    assert not score_answer(["a"], exact).strict
    unanswered = score_answer(["a", "b"], exact, answered=False)
    assert (unanswered.score, unanswered.strict) == (0.0, False)
    approx = Gold(answer="dry season", answer_type="string_approximate")
    assert score_answer("Dry Season", approx).strict
    assert score_answer("the dry season", approx).score == 0.0


# --- agreement with upstream ---------------------------------------------------


def _evaluator_root() -> Path:
    if not LOCAL.exists():
        pytest.skip(f"no local config at {LOCAL}")
    root = load_settings(LOCAL).kramabench.evaluator_root
    if not (root / "benchmark" / "metrics.py").exists():
        pytest.skip("KramaBench evaluator store not fetched")
    return root


@pytest.fixture(scope="module")
def upstream() -> types.ModuleType:
    root = _evaluator_root()
    path = root / "benchmark" / "metrics.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == scoring.UPSTREAM_METRICS_SHA256
    fixtures = root / "benchmark" / "fixtures" / "answer_type_fixtures.json"
    assert hashlib.sha256(fixtures.read_bytes()).hexdigest() == (
        scoring.UPSTREAM_ANSWER_TYPES_SHA256
    )

    class ExactJudge:  # stands in for the hosted paraphrase judge
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def evaluate_paraphrase(self, p: str, t: str) -> tuple[bool, int, int, int]:
            return p.strip().lower() == t.strip().lower(), 0, 0, 0

    stubs = {
        "nltk": types.ModuleType("nltk"),
        "rouge_score": types.ModuleType("rouge_score"),
        "rouge_score.rouge_scorer": types.ModuleType("rouge_score.rouge_scorer"),
        "benchmark": types.ModuleType("benchmark"),
        "benchmark.llm_tools": types.ModuleType("benchmark.llm_tools"),
    }
    stubs["rouge_score"].rouge_scorer = stubs["rouge_score.rouge_scorer"]  # type: ignore[attr-defined]
    for name in ("GPTInterface", "LLMInterface", "OllamaInterface"):
        setattr(stubs["benchmark.llm_tools"], name, ExactJudge)
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("kb_upstream_metrics", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


SCALARS: list[Any] = [
    0, 1, 42, -3, 0.0, 1.5, 3.1416, 3.14159, 1e-7, 100.0, 101.0, 99.95,
    "42", " 42 ", "12%", "1.5", "abc", "ABC ", "", "Utah", True, False, None, [1], ["a"],
]  # fmt: skip
LISTS: list[Any] = [
    [], ["a"], ["a", "b"], ["A ", "b"], ["b", "a", "c"], [1, 2], [1.0, 2.0], ["1", "2.0000001"],
    [1.005, 2], [1.02, 2.0], "a, b", '["a", "b"]', "['a', 'b']", "[1, 2]", "a", 5, None, [None],
    [["a"]], ["a", 1], [True],
]  # fmt: skip


def _same(a: float, b: float | None) -> bool:
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return False
    return a == pytest.approx(b, abs=0, rel=1e-12)


def test_success_and_rae_match_upstream(upstream: types.ModuleType) -> None:
    us, ur = upstream.Success(), upstream.RAEScore()
    for p, t in itertools.product(SCALARS, [0, 42, 0.0, 3.1416, 100.0, "Utah", "42", "12%"]):
        assert success(p, t) == us(p, t)[0], (p, t)
        if t in (0, 0.0, "Utah"):
            continue  # zero and non-numeric targets raise upstream; both give 0
        want = ur(p, t)[0]
        if isinstance(want, float) and math.isnan(want):
            assert rae_score(p, t) == 0.0  # deviation 4
        else:
            assert _same(rae_score(p, t), want), (p, t)


def test_f1_matches_upstream(upstream: types.ModuleType) -> None:
    uf, ua = upstream.F1(), upstream.F1Approximate()
    targets: list[Any] = [[], ["a"], ["a", "b"], [1, 2], [1.0, 2.0], ["a", 1], "a"]
    for p, t in itertools.product(LISTS, targets):
        assert _same(f1(p, t), uf(p, t)[0]), (p, t)
        if isinstance(t, list):
            # The stub judge is exact equality, which is deviation 2's rule.
            assert _same(f1_approximate(p, t), ua(p, t)[0]), (p, t)


def test_every_gold_answer_scores_full_marks() -> None:
    root = _evaluator_root()
    tasks = load_tasks(root)
    assert len([t for t in tasks.values() if "-tiny" not in t.workload]) == 104
    for t in tasks.values():
        s = score_answer(t.gold.answer, t.gold)
        assert (s.score, s.strict) == (1.0, True), t.key
