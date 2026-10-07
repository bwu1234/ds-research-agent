"""A synthetic KramaBench-shaped layout for offline tests (no benchmark data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ds_research_agent.config import Settings, load_settings
from eval.kramabench.tasks import DOMAINS

EXAMPLE = Path(__file__).parents[1] / "config" / "example.yaml"

ANSWERS: list[tuple[str, Any]] = [
    ("numeric_exact", 42),
    ("string_exact", "Utah"),
    ("list_exact", ["a", "b"]),
    ("numeric_approximate", 3.5),
]


def _task(domain: str, n: int) -> dict[str, Any]:
    diff = "easy" if n % 2 else "hard"
    at, ans = ANSWERS[(n - 1) % len(ANSWERS)]
    return {
        "id": f"{domain}-{diff}-{n}",
        "query": f"Synthetic question {n} about {domain}?",
        "answer": ans,
        "answer_type": at,
        "data_sources": [f"f{n}.csv"],
        "subtasks": [{"id": f"{domain}-{diff}-{n}-1", "query": "q", "answer": 1}],
    }


def make_layout(root: Path, per_domain: int = 4) -> tuple[Path, Path]:
    """Write ``evaluator/workload`` and ``visible/<domain>/input``; return both roots."""
    ev, vis = root / "evaluator", root / "visible"
    (ev / "workload").mkdir(parents=True)
    for d in DOMAINS:
        tasks = [_task(d, n) for n in range(1, per_domain + 1)]
        (ev / "workload" / f"{d}.json").write_text(json.dumps(tasks))
        inp = vis / d / "input"
        inp.mkdir(parents=True)
        for n in range(1, per_domain + 1):
            (inp / f"f{n}.csv").write_text(f"x,y\n{n},{n * 2}\n")
    # Smoke variants reuse a parent ID with different fields.
    legal = _task("legal", 2) | {"answer_type": "numeric_exact", "answer": 7}
    (ev / "workload" / "legal-tiny.json").write_text(json.dumps([legal]))
    (ev / "workload" / "environment-tiny.json").write_text(json.dumps([_task("environment", 1)]))
    # An unresolvable source keeps a task out of the D3 sample.
    wl = json.loads((ev / "workload" / "wildfire.json").read_text())
    wl[0]["data_sources"] = ["missing.csv"]
    (ev / "workload" / "wildfire.json").write_text(json.dumps(wl))
    return ev, vis


def settings_for(root: Path, **eval_overrides: Any) -> Settings:
    ev, vis = make_layout(root)
    s = load_settings(EXAMPLE)
    kb = s.kramabench.model_copy(
        update={"evaluator_root": ev, "visible_root": vis, "checkout": root / "repo"}
    )
    e = s.eval.model_copy(
        update={
            "ledger_path": root / "ledger.sqlite3",
            "bootstrap_resamples": 200,
            "sample_per_domain": 1,
            "split_sha256": None,  # the example config freezes the real split
            **eval_overrides,
        }
    )
    return s.model_copy(update={"kramabench": kb, "eval": e})
