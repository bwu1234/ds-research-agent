"""Submissions, their checks, and the reproduction comparator.

Three claims stay separate (docs/data-and-provenance.md): observed access
(the trusted audit of the fresh rerun), reproduction (the rerun prints the
submitted answer), and correctness (the evaluator's job, not this module's).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

# Frozen before any scored run; change only with a new name.
COMPARATOR = "reproduction-v1"
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Submission(_Frozen):
    answer: Any
    files_used: tuple[str, ...]
    program: str
    assumptions: tuple[str, ...] = ()


def check_submission(args: dict[str, Any], allowed_files: Sequence[str]) -> list[str]:
    """Problems beyond the JSON Schema; empty when the submission is accepted."""
    problems = []
    unknown = sorted(set(args.get("files_used", ())) - set(allowed_files))
    if unknown:
        problems.append(f"files_used lists files not given for this task: {', '.join(unknown)}")
    if not str(args.get("program", "")).strip():
        problems.append("program is empty")
    answer = args.get("answer")
    if isinstance(answer, float) and not math.isfinite(answer):
        problems.append("answer is not a finite number")
    if isinstance(answer, list) and not answer:
        problems.append("answer is an empty list")
    return problems


def program_answer(stdout: str) -> tuple[bool, Any]:
    """The value of ``{"answer": ...}`` on the last non-empty stdout line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False, None
    try:
        obj = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False, None
    if not isinstance(obj, dict) or "answer" not in obj:
        return False, None
    return True, obj["answer"]


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return False


def reproduces(submitted: Any, rerun: Any) -> bool:
    """``reproduction-v1``: same JSON shape and order; strings exactly equal;
    numbers equal within 1e-9 relative (1e-12 absolute) to absorb float
    formatting only. It says nothing about whether the answer is right."""
    return _same(submitted, rerun)
