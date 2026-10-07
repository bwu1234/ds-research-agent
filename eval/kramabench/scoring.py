"""Local deterministic KramaBench scoring (profile ``local-deterministic-v1``).

A port of the upstream comparators in ``benchmark/metrics.py`` at the pinned
commit, selected per answer type as ``benchmark/fixtures/
answer_type_fixtures.json`` does, with the hosted judges removed. The port
is checked against the upstream file itself in ``tests/test_scoring.py``
(upstream's module is loaded with its LLM imports stubbed), and the hashes
below pin which upstream text it follows.

Why a port and not an import: upstream ``metrics.py`` imports its OpenAI
judge client at module load and ``F1`` calls ``eval()`` on model output.

Deviations from upstream, all deliberate:

1. ``string_approximate``: upstream asks an LLM whether the strings are
   paraphrases. Here it is upstream ``Success`` (case- and edge-whitespace-
   insensitive equality, numeric if both sides parse as numbers).
   Paraphrases score zero.
2. ``list_approximate``: upstream ``F1Approximate`` asks an LLM per string
   element. Here string elements use the same equality as ``F1``; numeric
   elements keep upstream's 1% relative tolerance.
3. Model output that is a string where a list is expected is parsed with
   ``json.loads``, then ``ast.literal_eval``, then comma splitting. Upstream
   uses ``eval()`` in place of ``literal_eval``; results agree for Python
   literals, and expressions are never executed.
4. A non-finite answer score (``rae_score`` of a NaN prediction) is 0.
5. Upstream drops tasks whose evaluation raises; here every task in the
   manifest gets a row, and missing or malformed answers score 0.

Strict answer accuracy (not upstream): exact and approximate-string types
need ``success == 1``, list types need F1 == 1, and ``numeric_approximate``
needs ``|p - t| <= max(STRICT_ABS_TOL, STRICT_REL_TOL * |t|)``. The 1%
relative tolerance is upstream's own list-element threshold; it was fixed
before any model run.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from eval.kramabench.tasks import Gold

PROFILE = "local-deterministic-v1"
UPSTREAM_COMMIT = "b2e0d77540263f8b6119f977fe79a8c4386b5a02"
UPSTREAM_METRICS_SHA256 = "332502de61950c27a2273823851794db669dc885f656238a447178691457ce30"
UPSTREAM_ANSWER_TYPES_SHA256 = "b0b13c0ad9cde2830b788d40f47a1ee5f34b07fc1dd4ab4643eeefb0aa952c8e"

STRICT_REL_TOL = 0.01
STRICT_ABS_TOL = 1e-6
# Upstream F1Approximate's numeric element threshold.
LIST_APPROX_REL_TOL = 0.01

# Answer type -> metric used as the answer score.
METRICS = {
    "numeric_exact": "success",
    "string_exact": "success",
    "string_approximate": "success",
    "numeric_approximate": "rae_score",
    "list_exact": "f1",
    "list_approximate": "f1_approximate",
}


class AnswerScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = PROFILE
    answer_type: str
    metric: str
    score: float
    strict: bool


# --- upstream comparators -------------------------------------------------


def str_to_float(num_string: str) -> float:
    if num_string.endswith("%"):
        return float(num_string.strip("%")) / 100
    return float(num_string)


def try_convert_to_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            cleaned = value.strip()
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1].strip()
            return str_to_float(cleaned)
        except ValueError, TypeError:
            return value
    return value


def success(predicted: Any, target: Any) -> int:
    if predicted is None:
        return 0
    try:
        p = try_convert_to_number(predicted)
        t = try_convert_to_number(target)
        if isinstance(p, (int, float)) and isinstance(t, (int, float)):
            if p == t:
                return 1
            if isinstance(t, float) or isinstance(p, float):
                if t == 0:
                    return int(abs(p) < 1e-6)
                rea = abs(p - t) / abs(t)
                abs_diff = abs(p - t)
                larger = max(abs(p), abs(t))
                return int(rea < 1e-6 or abs_diff < 1e-6 or abs_diff < larger * 1e-4)
            return 0
        if isinstance(target, str) and isinstance(predicted, str):
            return int(predicted.strip().lower() == target.strip().lower())
        if isinstance(target, str) and isinstance(predicted, (int, float)):
            return 0
        return int(predicted == target)
    except Exception:
        return 0


def rae_score(predicted: Any, target: Any) -> float:
    try:
        p = str_to_float(predicted) if isinstance(predicted, str) else predicted
        t = str_to_float(target) if isinstance(target, str) else target
        rae = abs(p - t) / abs(t)
        score = 1 / (1 + rae)
    except Exception:
        return 0.0
    return float(score) if math.isfinite(score) else 0.0  # deviation 4


def _parse_list(predicted: str) -> Any:
    """Deviation 3: ``eval`` replaced by ``ast.literal_eval``."""
    try:
        return json.loads(predicted)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(predicted)
    except Exception:
        return [item.strip() for item in predicted.split(",") if item.strip()]


def f1(predicted: Any, target: Any) -> float:
    try:
        if isinstance(predicted, list) and isinstance(target, str):
            target = json.loads(target)
        if isinstance(predicted, str) and isinstance(target, list):
            predicted = _parse_list(predicted)
        if isinstance(predicted, str) and isinstance(target, str):
            return float(predicted.strip().lower() == target.strip().lower())
        if len(target) == 0:
            return float(len(predicted) == 0)
        matched: set[int] = set()
        recall_cnt = 0
        for t in target:
            for j, p in enumerate(predicted):
                if isinstance(t, str):
                    if t.strip().lower() == str(p).strip().lower():
                        matched.add(j)
                        recall_cnt += 1
                        break
                elif isinstance(t, (float, int)):
                    if isinstance(p, (float, int)):
                        pp = p
                    elif isinstance(p, str):
                        try:
                            pp = str_to_float(p)
                        except Exception:
                            continue
                    else:
                        continue
                    if abs(pp - t) / max(abs(t), 1e-12) < 1e-6:
                        matched.add(j)
                        recall_cnt += 1
                        break
                elif p == t:
                    matched.add(j)
                    recall_cnt += 1
                    break
        return _f1(recall_cnt, len(target), len(matched), len(predicted))
    except Exception:
        return 0.0


def f1_approximate(predicted: Any, target: Any) -> float:
    try:
        if isinstance(target, str):
            target = json.loads(target)
        if isinstance(predicted, str):
            try:
                predicted = json.loads(predicted)
            except json.JSONDecodeError:
                predicted = ast.literal_eval(predicted)  # deviation 3
        predicted = list(predicted)
        if len(target) == 0:
            return float(len(predicted) == 0)
        matched: set[int] = set()
        recall_cnt = 0
        for t in target:
            for j, p in enumerate(predicted):
                if isinstance(t, str):
                    # Deviation 2: equality in place of the LLM paraphrase judge.
                    if str(p).strip().lower() == t.strip().lower():
                        matched.add(j)
                        recall_cnt += 1
                        break
                else:
                    if isinstance(p, (float, int)):
                        pp = p
                    elif isinstance(p, str):
                        try:
                            pp = str_to_float(p)
                        except Exception:
                            continue
                    else:
                        continue
                    if abs(pp - t) / max(abs(t), 1e-12) <= LIST_APPROX_REL_TOL:
                        matched.add(j)
                        recall_cnt += 1
                        break
        return _f1(recall_cnt, len(target), len(matched), len(predicted))
    except Exception:
        return 0.0


def _f1(recall_cnt: int, n_target: int, n_matched: int, n_pred: int) -> float:
    recall = recall_cnt / n_target
    precision = n_matched / n_pred if n_pred else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- profile --------------------------------------------------------------


def within_tolerance(predicted: Any, target: Any) -> bool:
    try:
        p = str_to_float(predicted) if isinstance(predicted, str) else predicted
        t = str_to_float(target) if isinstance(target, str) else target
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return False
        diff = abs(p - t)
        return bool(math.isfinite(diff) and diff <= max(STRICT_ABS_TOL, STRICT_REL_TOL * abs(t)))
    except Exception:
        return False


def score_answer(predicted: Any, gold: Gold, *, answered: bool = True) -> AnswerScore:
    """Score one answer. ``answered=False`` (no answer, malformed output,
    timeout, crash) scores zero without consulting the comparators, so a
    JSON ``null`` answer and a missing answer stay distinguishable upstream."""
    at = gold.answer_type
    metric = METRICS[at]
    if not answered:
        return AnswerScore(answer_type=at, metric=metric, score=0.0, strict=False)
    if metric == "success":
        s = float(success(predicted, gold.answer))
        strict = s == 1.0
    elif metric == "rae_score":
        s = rae_score(predicted, gold.answer)
        strict = within_tolerance(predicted, gold.answer)
    elif metric == "f1":
        s = f1(predicted, gold.answer)
        strict = s == 1.0
    else:
        s = f1_approximate(predicted, gold.answer)
        strict = s == 1.0
    return AnswerScore(answer_type=at, metric=metric, score=s, strict=strict)
