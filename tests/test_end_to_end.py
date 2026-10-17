"""The end_to_end condition offline: synthetic catalogue, scripted model, fake
search and workspace; selection, mounts, scoring, and a replay from the ledger."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.agent import tools as given_tools
from ds_research_agent.agent.discovery import CatalogueTools
from ds_research_agent.agent.end_to_end import (
    RESTARTED,
    TOOLS,
    CatalogueAccess,
    CatalogueMismatch,
)
from ds_research_agent.catalogue import Manifest, load_manifest
from ds_research_agent.catalogue.profiler import build_catalogue
from ds_research_agent.ledger import Ledger
from ds_research_agent.models import ChatMessage, ChatResult, ToolCall, ToolSpec, Usage
from ds_research_agent.retrieval.mcp_adapter import SearchResponse
from ds_research_agent.sandbox import (
    AuditResult,
    CellResult,
    InputMount,
    ObservedRead,
    SandboxRun,
    SessionEnd,
)
from eval.kramabench import agent_runs, discovery_agent, end_to_end, failures, report
from eval.kramabench.baselines import Harness, ReplayClient, run_batch
from eval.kramabench.run import batch_sections, new_batch
from eval.kramabench.split import split_for
from tests.kramabench_synthetic import settings_for

# The given-files prefix as recorded on its batches (0404-dev-given_files and
# the D3 batches since (f)): the end-to-end prompt is built from its parts.
GIVEN_FILES_PREFIX = "662c04e7b189089c655d7664f7591efb5bcfe8dc1fd4c0c46c7c1ab45dcb8184"


def test_given_files_prefix_is_unchanged() -> None:
    assert agent_runs.prefix_sha256() == GIVEN_FILES_PREFIX
    assert given_tools.WORK_AND_FINISH in given_tools.SYSTEM_PROMPT
    assert end_to_end.prefix_sha256() != GIVEN_FILES_PREFIX


def _audit(reads: Sequence[str] = ()) -> AuditResult:
    return AuditResult(
        complete=True,
        issues=[],
        processes=1,
        data_reads=sorted(reads),
        data_dirs_listed=[],
        failed_data_opens=[],
        blocked_escapes=[],
        unparsed_lines=0,
    )


class FakeWorkspace:
    """Records mounts; cells succeed; the rerun reads ``reads`` and prints ``answer``."""

    def __init__(self, answer: Any, reads: Callable[[], Sequence[str]]) -> None:
        self.answer, self.reads = answer, reads
        self.mounts: list[list[InputMount]] = []

    @property
    def session_id(self) -> str | None:
        return "dsra-krn-test"

    async def set_inputs(self, inputs: Sequence[InputMount]) -> None:
        self.mounts.append(list(inputs))

    async def execute(self, code: str, timeout_s: int) -> CellResult:
        return CellResult(
            cell=1, code=code, status="ok", traceback=None, stdout="ok", stderr="",
            stdout_truncated=False, stderr_truncated=False, wall_s=0.01, audit=_audit(),
            observed_reads=[],
        )  # fmt: skip

    async def rerun(self, program: str) -> SandboxRun:
        reads = list(self.reads())
        return SandboxRun(
            container="dsra-sbx-test", image_id="sha256:x", exit_code=0, timed_out=False,
            wall_s=0.5, stdout=json.dumps({"answer": self.answer}) + "\n", stderr="",
            stdout_truncated=False, stderr_truncated=False, audit=_audit(reads),
            observed_reads=[ObservedRead(container_path=r, host_path=Path("/h")) for r in reads],
            scratch_dir=Path("/s"), inputs=tuple(reads),
        )  # fmt: skip

    async def close(self) -> list[SessionEnd]:
        return []


class OneCardSearcher:
    """Returns the file card of f1.csv in the filtered domain (collapsed, its group)."""

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
        return _response(query, corpus, [f"{domain}/input/f1.csv.md"])


def _response(query: str, corpus: str, docs: Sequence[str]) -> SearchResponse:
    return SearchResponse.model_validate(
        {
            "query": query,
            "corpora": [corpus],
            "returned": len(docs),
            "results": [
                {"rank": i + 1, "score": 0.9, "chunk_id": f"{d}::0", "document_id": d,
                 "source": d, "text": "x y"}
                for i, d in enumerate(docs)
            ],
        }
    )  # fmt: skip


def _reply(name: str, args: dict[str, Any]) -> ChatResult:
    return ChatResult(
        message=ChatMessage(
            role="assistant", content="", tool_calls=(ToolCall(name=name, arguments=args),)
        ),
        usage=Usage(),
        done_reason="stop",
        wall_s=0.1,
    )


class Analyst:
    """Reads the group card, selects its member f<n>, tries an undiscovered ID,
    then selects the whole group (a restart, since a cell ran) and submits."""

    def __init__(self, gid: str, n: int, domain: str, answer: Any) -> None:
        self.gid, self.n, self.answer = gid, n, answer
        self.path = f"/data/{domain}/input/f{n}.csv"
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        assert tuple(tools) == TOOLS
        self.requests.append(list(messages))
        turn = sum(m.role == "assistant" for m in messages)
        if turn == 0:
            return _reply("read_card", {"id": self.gid, "members": [f"*/f{self.n}.csv"]})
        if turn == 1:
            sel = [{"id": self.gid, "members": [f"*/f{self.n}.csv"]}]
            return _reply("run_python", {"code": "x = 1", "select": sel})
        if turn == 2:
            return _reply("run_python", {"code": "x", "select": [{"id": "legal/nope"}]})
        if turn == 3:
            return _reply("run_python", {"code": "y = 2", "select": [{"id": self.gid}]})
        # The file named by its catalogue path (as the prompt lists it) and its mount.
        used = [self.path.removeprefix("/data/"), self.path]
        return _reply(
            "submit_answer", {"answer": self.answer, "files_used": used, "program": "print(1)"}
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


def _group_id(manifest: Manifest, domain: str) -> str:
    (g,) = [g for g in manifest.groups if g.domain == domain]
    return g.group_id


def test_end_to_end_selects_mounts_scores_and_replays(harness: Harness) -> None:
    h = harness
    manifest = load_manifest(h.settings.catalogue.manifest_path)
    key = next(k for k in h.split.dev if h.tasks[k].domain == "legal")
    task = h.tasks[key]
    n = int(task.query.split()[2])
    gid = _group_id(manifest, "legal")
    batch = new_batch(
        h.settings, h.model, h.split, condition="end_to_end", task_set="dev", keys=(key,),
        repeats=1, batch_id="e2e-test",
    )  # fmt: skip
    assert batch.system_prompt_sha256 == end_to_end.prefix_sha256()
    assert "end_to_end" in batch_sections("end_to_end")
    h.ledger.add_batch(batch)
    searcher = OneCardSearcher()
    model = Analyst(gid, n, "legal", task.gold.answer)
    ws = FakeWorkspace(task.gold.answer, lambda: [model.path])

    async def run_task(hh: Harness, b: Any, t: Any, r: int, c: Any) -> Any:
        return await end_to_end.run_end_to_end_task(
            hh, b, t, r, c, lambda *_: ws, "sha256:x", searcher, manifest
        )

    asyncio.run(run_batch(h, batch, lambda _t, _r: model, agent_task=run_task))
    (run,) = h.ledger.runs(batch.batch_id)
    assert run.stop_reason == "submitted", run.error
    assert run.access_policy == "catalogue_selected_domain"
    assert run.input_manifest["seed_cards"] == 10 and run.input_manifest["domain_filter"] == "legal"

    # The seed search is in the prompt, and nothing is mounted until selected.
    user = model.requests[0][1].content
    assert gid in user and "No data files are mounted yet" in user
    # Mounts: f<n> alone, then the whole group (selection so far, manifest order).
    host = h.settings.catalogue.data_root
    assert [[m.container_path for m in ms] for ms in ws.mounts] == [
        [model.path],
        [f"/data/legal/input/f{i}.csv" for i in range(1, 5)],
    ]
    assert ws.mounts[0][0].host_path == host / "legal" / "input" / f"f{n}.csv"

    tool_texts = [m.content for m in model.requests[-1] if m.role == "tool"]
    assert "Members matching" in tool_texts[0]  # read_card with globs
    assert (
        tool_texts[1].startswith("[select: mounted 1 more file(s)")
        and RESTARTED not in tool_texts[1]
    )
    assert "was not returned by a search" in tool_texts[2] and "not run" in tool_texts[2]
    assert "mounted 3 more file(s)" in tool_texts[3] and RESTARTED in tool_texts[3]

    steps = h.ledger.steps(run.run_id)
    assert [s.kind for s in steps] == [
        "search", "model", "model", "selection", "model", "selection", "model", "selection",
        "model",
    ]  # fmt: skip
    first = steps[3].response
    assert first is not None and first["added"][0]["sha256"].startswith("sha256:")
    assert steps[5].response and steps[5].response["problems"]

    # Accepted first time, recorded and audited by mount path.
    sub = h.ledger.submission(run.run_id)
    assert sub is not None and sub.files_used == [model.path]
    score = h.ledger.score(run.run_id, "local-deterministic-v1")
    assert score is not None and score.strict and score.verified_success

    summary = report.summarise(h.ledger, batch, h.tasks, 200, 0)
    d = summary["agent"]["discovery"]
    assert d["selection_recall"] == 1.0 and d["selection_complete"] == 1.0
    assert d["selection_precision"] == 0.25 and d["seed_complete"] == 1.0
    assert d["selections_per_run"] == 3.0 and d["refused_selections"] == 1
    assert summary["agent"]["model_requests_per_run"] == [5]

    # Replay from the ledger: search, model, and programs; selections resolved again.
    replay = new_batch(
        h.settings, h.model, h.split, condition="end_to_end", task_set="dev", keys=(key,),
        repeats=1, batch_id="e2e-replay", replay_of=batch,
    )  # fmt: skip
    h.ledger.add_batch(replay)
    recorded = h.ledger.steps(run.run_id)
    rws = agent_runs.ReplayWorkspace(h.ledger.programs(run.run_id))

    async def replay_task(hh: Harness, b: Any, t: Any, r: int, c: Any) -> Any:
        rs = discovery_agent.ReplaySearcher(recorded)
        return await end_to_end.run_end_to_end_task(
            hh, b, t, r, c, lambda *_: rws, "sha256:x", rs, manifest
        )

    model_steps = [s for s in recorded if s.kind == "model"]
    asyncio.run(
        run_batch(
            h, replay, lambda _t, _r: ReplayClient(h.model, model_steps), agent_task=replay_task
        )
    )
    (again,) = h.ledger.runs("e2e-replay")
    assert h.ledger.answer(again.run_id) == h.ledger.answer(run.run_id).model_copy(  # type: ignore[union-attr]
        update={"run_id": again.run_id}
    )
    v0, v1 = h.ledger.verification(run.run_id), h.ledger.verification(again.run_id)
    assert v0 and v1 and v0.model_dump(exclude={"run_id"}) == v1.model_dump(exclude={"run_id"})
    assert len(searcher.queries) == 1  # the replay searched nothing


def test_submission_must_use_selected_files(harness: Harness) -> None:
    h = harness
    manifest = load_manifest(h.settings.catalogue.manifest_path)
    key = next(k for k in h.split.dev if h.tasks[k].domain == "legal")
    task = h.tasks[key]
    path = "/data/legal/input/f1.csv"

    class SubmitsUnselected:
        async def chat(
            self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
        ) -> ChatResult:
            return _reply(
                "submit_answer", {"answer": task.gold.answer, "files_used": [path], "program": "p"}
            )

    batch = new_batch(
        h.settings, h.model, h.split, condition="end_to_end", task_set="dev", keys=(key,),
        repeats=1, batch_id="e2e-unselected",
    )  # fmt: skip
    h.ledger.add_batch(batch)
    ws = FakeWorkspace(task.gold.answer, lambda: [path])
    model = SubmitsUnselected()

    async def run_task(hh: Harness, b: Any, t: Any, r: int, c: Any) -> Any:
        return await end_to_end.run_end_to_end_task(
            hh, b, t, r, c, lambda *_: ws, None, OneCardSearcher(), manifest
        )

    asyncio.run(run_batch(h, batch, lambda _t, _r: model, agent_task=run_task))
    (run,) = h.ledger.runs(batch.batch_id)
    # Every submission is refused, so the run spends its turns and ends unanswered.
    assert run.stop_reason == "max_steps" and ws.mounts == []
    last = h.ledger.steps(run.run_id)[-1].request["messages"][-1]["content"]
    assert f"files_used lists files not selected in this run: {path}" in last


def _access(h: Harness, searcher: Any, ws: FakeWorkspace) -> tuple[CatalogueAccess, Any]:
    manifest = load_manifest(h.settings.catalogue.manifest_path)
    tools = CatalogueTools(
        searcher, manifest, h.settings.catalogue.cards_dir, h.settings.catalogue.corpus,
        {"equals": {"domain": "legal"}}, h.settings.discovery,
    )  # fmt: skip
    assert h.settings.end_to_end is not None
    return CatalogueAccess(tools, ws, h.settings.catalogue.data_root, h.settings.end_to_end), tools


def test_a_changed_file_stops_the_run(harness: Harness) -> None:
    h = harness
    ws = FakeWorkspace(0, list)
    access, tools = _access(h, OneCardSearcher(), ws)
    asyncio.run(tools.search("q"))
    (h.settings.catalogue.data_root / "legal" / "input" / "f1.csv").write_text("x,y\n9,9\n")
    gid = tools.discovered.ids[0]
    with pytest.raises(CatalogueMismatch):
        asyncio.run(access.select({"select": [{"id": gid, "members": ["*/f1.csv"]}]}))
    assert ws.mounts == []


def test_seed_limit_keeps_the_first_results(harness: Harness) -> None:
    h = harness
    s = h.settings.discovery.model_copy(update={"collapse_groups": False})

    class TwoCards:
        async def search(self, query: str, **_: Any) -> SearchResponse:
            return _response(query, "c", ["legal/input/f2.csv.md", "legal/input/f1.csv.md"])

    manifest = load_manifest(h.settings.catalogue.manifest_path)
    tools = CatalogueTools(TwoCards(), manifest, h.settings.catalogue.cards_dir, "c", None, s)
    text = asyncio.run(tools.search("q", limit=1))
    (only,) = tools.discovered.ids
    assert manifest.by_catalogue_id()[only].file_path == "legal/input/f2.csv"
    assert "(1):" in text and "f1.csv" not in text


class Scripted:
    """Replies from a list, one per model turn; records each request."""

    def __init__(self, *replies: ChatResult) -> None:
        self.replies = list(replies)
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        self.requests.append(list(messages))
        return self.replies.pop(0)


def _run_one(h: Harness, model: Any, searcher: Any, batch_id: str) -> Any:
    manifest = load_manifest(h.settings.catalogue.manifest_path)
    key = next(k for k in h.split.dev if h.tasks[k].domain == "legal")
    batch = new_batch(
        h.settings, h.model, h.split, condition="end_to_end", task_set="dev", keys=(key,),
        repeats=1, batch_id=batch_id,
    )  # fmt: skip
    h.ledger.add_batch(batch)
    ws = FakeWorkspace(0, list)

    async def run_task(hh: Harness, b: Any, t: Any, r: int, c: Any) -> Any:
        return await end_to_end.run_end_to_end_task(
            hh, b, t, r, c, lambda *_: ws, None, searcher, manifest
        )

    asyncio.run(run_batch(h, batch, lambda _t, _r: model, agent_task=run_task))
    return batch


def test_repeated_refused_selections_ask_for_a_new_plan(harness: Harness) -> None:
    h = harness
    bad = [
        _reply("run_python", {"code": "x", "select": [{"id": f"legal/nope{i}"}]}) for i in (1, 2)
    ]
    model = Scripted(*bad, _reply("submit_answer", {"answer": 1, "files_used": [], "program": "p"}))
    _run_one(h, model, OneCardSearcher(), "e2e-refused")
    tool = [m.content for m in model.requests[-1] if m.role == "tool"]
    # Different IDs, same kind of error: the second refusal asks for a re-plan.
    assert "same error has now occurred" not in tool[0]
    assert "The same error has now occurred 2 times" in tool[1]


def test_empty_seed_search_says_nothing_can_be_selected(harness: Harness) -> None:
    h = harness

    class Nothing:
        async def search(self, query: str, **kw: Any) -> SearchResponse:
            return _response(query, kw["corpus"], [])

    model = Scripted(_reply("submit_answer", {"answer": 1, "files_used": [], "program": "p"}))
    batch = _run_one(h, model, Nothing(), "e2e-empty")
    user = model.requests[0][1].content
    assert "returned no results, so no data files can be selected" in user
    assert "Try other terms" not in user and "select the ones you need" not in user
    d = report.summarise(h.ledger, batch, h.tasks, 200, 0)["agent"]["discovery"]
    assert d["empty_seed"] == 1 and d["seed_complete"] == 0.0


def test_seed_reach_follows_collapse_and_the_runs_catalogue(harness: Harness) -> None:
    h = harness
    manifest = load_manifest(h.settings.catalogue.manifest_path)
    model = Scripted(_reply("submit_answer", {"answer": 1, "files_used": [], "program": "p"}))
    batch = _run_one(h, model, OneCardSearcher(), "e2e-seed")
    (run,) = h.ledger.runs(batch.batch_id)
    collapsed = end_to_end.seed_files(h.ledger, run.run_id, manifest, 10, collapse_groups=True)
    single = end_to_end.seed_files(h.ledger, run.run_id, manifest, 10, collapse_groups=False)
    assert len(collapsed) == 4 and single == ["legal/input/f1.csv"]

    root = h.settings.kramabench.visible_root
    same = end_to_end.summary_extra(h.ledger, batch, h.tasks, root, manifest)
    assert same["manifest_mismatch"] == 0 and same["seed_complete"] is not None
    rebuilt = manifest.model_copy(
        update={"entries": [e.model_copy(update={"card_hash": "x"}) for e in manifest.entries]}
    )
    other = end_to_end.summary_extra(h.ledger, batch, h.tasks, root, rebuilt)
    assert other["manifest_mismatch"] == 1 and other["seed_complete"] is None


@pytest.mark.parametrize(
    ("row", "state"),
    [
        ({"complete": 1.0, "seed_complete": 1.0}, "selection_complete"),
        ({"complete": 0.0, "seed_complete": 1.0}, "selection_incomplete"),
        ({"complete": 0.0, "seed_complete": 0.0}, "seed_incomplete"),
        ({"complete": 0.0}, "unscored"),  # recorded on another catalogue
        ({}, "unscored"),  # gold set did not resolve
    ],
)
def test_selection_state(row: dict[str, float], state: str) -> None:
    assert report._selection_state(row) == state


def test_show_places_a_run_by_how_far_its_gold_files_got(harness: Harness) -> None:
    h = harness

    class Nothing:
        async def search(self, query: str, **kw: Any) -> SearchResponse:
            return _response(query, kw["corpus"], [])

    model = Scripted(_reply("submit_answer", {"answer": 1, "files_used": [], "program": "p"}))
    batch = _run_one(h, model, Nothing(), "e2e-show")
    (run,) = h.ledger.runs(batch.batch_id)
    sel = report.run_selection(h.ledger, run.run_id, h.tasks)
    assert sel["state"] == "seed_incomplete" and sel["extra_selected"] == []
    assert sel["not_selected"] == sel["not_in_seed"] and sel["not_in_seed"]
    text = failures.render_run(h.ledger, run, h.tasks[run.task_key], selection=sel)
    assert "## selection (evaluator only): seed_incomplete" in text
    assert "resolution:" not in text  # end-to-end runs record no labelled inputs


def test_by_selection_pairs_each_run_with_the_base_on_its_task(harness: Harness) -> None:
    h = harness

    class Nothing:
        async def search(self, query: str, **kw: Any) -> SearchResponse:
            return _response(query, kw["corpus"], [])

    def submit() -> Scripted:
        return Scripted(_reply("submit_answer", {"answer": 1, "files_used": [], "program": "p"}))

    base = _run_one(h, submit(), OneCardSearcher(), "e2e-base")
    other = _run_one(h, submit(), Nothing(), "e2e-other")
    out = report.by_selection(h.ledger, [base], [other], h.tasks, 200, 0)
    s = out["seed_incomplete"]
    assert (s["runs"], s["parents"], s["diff"], s["diff_ci95"]) == (1, 1, 0.0, [0.0, 0.0])
    assert all(out[k]["runs"] == 0 for k in report.SELECTION_STATES if k != "seed_incomplete")
    assert report.by_selection(h.ledger, [base], [other], h.tasks, 200, 0, keys=[])[
        "seed_incomplete"
    ] == {"runs": 0}
