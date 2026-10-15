"""The discovery_agent condition end to end, offline: synthetic catalogue,
scripted model, fake search; then a replay from the ledger alone."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.catalogue.profiler import build_catalogue
from ds_research_agent.ledger import Ledger
from ds_research_agent.models import ChatMessage, ChatResult, ToolCall, ToolSpec, Usage
from ds_research_agent.retrieval.mcp_adapter import SearchResponse
from eval.kramabench import discovery_agent
from eval.kramabench.baselines import Harness, ReplayClient, ReplayMismatch, run_batch
from eval.kramabench.run import new_batch
from eval.kramabench.split import split_for
from tests.kramabench_synthetic import settings_for


def _reply(name: str, args: dict[str, Any]) -> ChatResult:
    return ChatResult(
        message=ChatMessage(
            role="assistant", content="", tool_calls=(ToolCall(name=name, arguments=args),)
        ),
        usage=Usage(),
        done_reason="stop",
        wall_s=0.1,
    )


class Chooser:
    """Searches once, then chooses the group found, narrowed to file n (or,
    for ``wrong``, the whole group)."""

    def __init__(self, wrong: set[int]) -> None:
        self.wrong = wrong

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        if len(messages) == 2:
            return _reply("search_catalogue", {"query": "x and y"})
        n = int(re.search(r"question (\d+)", messages[1].content).group(1))  # type: ignore[union-attr]
        gid = re.search(r"\] (\S+/g-\w+):", messages[-1].content).group(1)  # type: ignore[union-attr]
        members = [] if n in self.wrong else [f"*/f{n}.csv"]
        return _reply("choose_files", {"selections": [{"id": gid, "members": members}]})


class DomainSearcher:
    """Returns the file card of f1.csv in the filtered domain."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        *,
        corpus: str,
        top_k: int,
        max_chars: int,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        self.queries.append(query)
        domain = (filters or {})["equals"]["domain"]
        doc = f"{domain}/input/f1.csv.md"
        return SearchResponse.model_validate(
            {
                "query": query,
                "corpora": [corpus],
                "returned": 1,
                "results": [
                    {
                        "rank": 1,
                        "score": 0.9,
                        "chunk_id": f"{doc}::0",
                        "document_id": doc,
                        "source": doc,
                        "text": "x y",
                    }
                ],
            }
        )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    s = settings_for(tmp_path)
    cat = s.catalogue.model_copy(
        update={
            "data_root": s.kramabench.visible_root,
            "cards_dir": tmp_path / "cards",
            "manifest_path": tmp_path / "manifest.json",
            "corpus": "synthetic-cards",
        }
    )
    s = s.model_copy(update={"catalogue": cat})
    split, tasks = split_for(s)
    s = s.model_copy(update={"eval": s.eval.model_copy(update={"split_sha256": split.sha256()})})
    build_catalogue(s.catalogue)
    return Harness(s, s.model, Ledger(s.eval.ledger_path), split, tasks, "rev-test")


def test_discovery_runs_score_summarise_and_replay(harness: Harness) -> None:
    h = harness
    from ds_research_agent.catalogue import load_manifest

    manifest = load_manifest(h.settings.catalogue.manifest_path)
    keys = tuple(k for k in h.split.dev if h.tasks[k].domain == "legal")[:3]
    assert len(keys) >= 2
    n_of = {k: int(h.tasks[k].query.split()[2]) for k in keys}
    batch = new_batch(
        h.settings, h.model, h.split, condition="discovery_agent", task_set="dev", keys=keys,
        repeats=1, batch_id="disc-test",
    )  # fmt: skip
    assert batch.scoring_profile == "discovery-v1"
    assert batch.system_prompt_sha256 == discovery_agent.prefix_sha256()
    h.ledger.add_batch(batch)
    searcher = DomainSearcher()
    client = Chooser(wrong={n_of[keys[1]]})

    async def task(hh: Harness, b: Any, t: Any, r: int, c: Any) -> Any:
        return await discovery_agent.run_discovery_task(hh, b, t, r, c, searcher, manifest)

    asyncio.run(run_batch(h, batch, lambda _t, _r: client, agent_task=task))
    runs = {r.task_key: r for r in h.ledger.runs(batch.batch_id)}
    assert {r.stop_reason for r in runs.values()} == {"chosen"}
    r0 = runs[keys[0]]
    assert r0.access_policy == "catalogue_search_domain"
    assert r0.input_manifest["domain_filter"] == "legal" and r0.input_manifest["groups"] == 6
    kinds = [s.kind for s in h.ledger.steps(r0.run_id)]
    assert kinds == ["model", "search", "model"]
    a0 = h.ledger.answer(r0.run_id)
    assert a0 and a0.value["files"] == [f"legal/input/f{n_of[keys[0]]}.csv"]

    scores = {k: h.ledger.score(runs[k].run_id, "discovery-v1") for k in keys}
    assert all(s is not None for s in scores.values())
    assert (scores[keys[0]].score, scores[keys[0]].strict) == (1.0, True)  # type: ignore[union-attr]
    # The whole 4-member group: gold found, three extra files.
    assert (scores[keys[1]].score, scores[keys[1]].strict) == (1.0, True)  # type: ignore[union-attr]
    rows = {r["task"]: r for r in discovery_agent.summary(h.ledger, batch, h.tasks,
            h.settings.kramabench.visible_root)["tasks"]}  # fmt: skip
    assert rows[keys[1]]["excess"] == 3.0 and rows[keys[1]]["precision"] == 0.25
    assert rows[keys[0]]["searches"] == 1.0 and rows[keys[0]]["model_requests"] == 2.0

    # Replay from the ledger: model steps to ReplayClient, searches to ReplaySearcher.
    replay = new_batch(
        h.settings, h.model, h.split, condition="discovery_agent", task_set="dev", keys=keys,
        repeats=1, batch_id="disc-replay", replay_of=batch,
    )  # fmt: skip
    h.ledger.add_batch(replay)
    steps_of = {k: h.ledger.steps(runs[k].run_id) for k in keys}

    async def replay_task(hh: Harness, b: Any, t: Any, r: int, c: Any) -> Any:
        rs = discovery_agent.ReplaySearcher(steps_of[t.key])
        return await discovery_agent.run_discovery_task(hh, b, t, r, c, rs, manifest)

    model_steps = {k: [s for s in v if s.kind == "model"] for k, v in steps_of.items()}
    asyncio.run(
        run_batch(
            h, replay, lambda t, _r: ReplayClient(h.model, model_steps[t.key]),
            agent_task=replay_task,
        )
    )  # fmt: skip
    for a, b in zip(h.ledger.runs(batch.batch_id), h.ledger.runs("disc-replay"), strict=True):
        aa, ab = h.ledger.answer(a.run_id), h.ledger.answer(b.run_id)
        assert aa and ab and aa.value == ab.value
    assert len(searcher.queries) == len(keys)  # the replay searched nothing


def test_replay_searcher_refuses_a_changed_search(harness: Harness) -> None:
    from ds_research_agent.ledger import Step

    step = Step(
        run_id="r",
        seq=1,
        kind="search",
        request_sha256="0" * 64,
        request={},
        response=None,
    )
    rs = discovery_agent.ReplaySearcher([step])
    with pytest.raises(ReplayMismatch):
        asyncio.run(rs.search("q", corpus="c", top_k=20, max_chars=200))
