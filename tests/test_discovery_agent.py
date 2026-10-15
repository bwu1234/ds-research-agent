"""Offline checks of the agentic-discovery loop: scripted model, fake search."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ds_research_agent.agent.discovery import (
    LAST_TURN,
    SYSTEM_PROMPT,
    TOOLS,
    CatalogueTools,
    run_discovery,
    user_prompt,
)
from ds_research_agent.catalogue import CatalogueEntry, CatalogueGroup, Manifest
from ds_research_agent.config import AgentSettings, DiscoverySettings
from ds_research_agent.models import ChatMessage, ChatResult, ToolCall, ToolSpec, Usage
from ds_research_agent.retrieval.mcp_adapter import SearchResponse

AGENT = AgentSettings(
    tool_call_max_repairs=1,
    max_steps=6,
    max_wall_s=1000,
    max_output_tokens=4096,
    max_tool_output_chars=50,
    replan_after_repeats=2,
    no_progress_after_repeats=0,
    no_progress_similarity=0.95,
    budget_warning_steps=0,
    submit_checks=0,
    server_retries=0,
)
SETTINGS = DiscoverySettings(
    max_steps=5,
    max_wall_s=1000,
    search_passages=20,
    snippet_chars=40,
    max_card_chars=200,
    max_listed_members=2,
    collapse_groups=True,
    budget_warning_steps=0,
)
SAT = [f"d/sat/sat-{y}.csv" for y in (2014, 2015, 2016)]


def entry(path: str) -> CatalogueEntry:
    return CatalogueEntry(
        catalogue_id=f"id:{path}",
        domain="d",
        file_path=path,
        size=10,
        file_hash="sha256:0",
        format="csv",
        document_id=f"{path}.md",
        card_hash="sha256:0",
        profiler_version=3,
        parse_status="ok",
    )


GROUP = CatalogueGroup(
    group_id="d/g-sat",
    domain="d",
    rules=("pattern",),
    member_ids=tuple(f"id:{p}" for p in SAT),
    directory="d/sat",
    format="csv",
    document_id="d/_groups/g-sat.md",
    card_hash="sha256:0",
    profiler_version=3,
)
MANIFEST = Manifest(
    profiler_version=3,
    corpus="c",
    entries=[entry(p) for p in [*SAT, "d/codes.csv", "d/other.csv"]],
    skipped=[],
    groups=[GROUP],
)


@pytest.fixture
def cards(tmp_path: Path) -> Path:
    for e in MANIFEST.entries:
        (tmp_path / e.document_id).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / e.document_id).write_text(f"# {e.file_path}\n\ncolumns: a, b\n")
    (tmp_path / GROUP.document_id).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / GROUP.document_id).write_text("# sat-#.csv (d, group of 3 files)\n\n" + "x" * 300)
    return tmp_path


def response(*docs: str) -> SearchResponse:
    return SearchResponse.model_validate(
        {
            "query": "q",
            "corpora": ["c"],
            "returned": len(docs),
            "results": [
                {
                    "rank": i,
                    "score": 0.5,
                    "chunk_id": f"{d}::0",
                    "document_id": d,
                    "source": d,
                    "text": f"passage   about {d}",
                }
                for i, d in enumerate(docs, 1)
            ],
        }
    )


class FakeSearcher:
    def __init__(self, *responses: SearchResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any] | None, int]] = []

    async def search(
        self,
        query: str,
        *,
        corpus: str,
        top_k: int,
        max_chars: int,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        self.calls.append((query, filters, top_k))
        return self.responses.pop(0)


def reply(*calls: tuple[str, dict[str, Any]], done: str = "stop") -> ChatResult:
    return ChatResult(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=tuple(ToolCall(name=n, arguments=a) for n, a in calls),
        ),
        usage=Usage(),
        done_reason=done,
        wall_s=1.0,
    )


class Script:
    def __init__(self, *replies: ChatResult) -> None:
        self.replies = list(replies)
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        assert tuple(tools) == TOOLS
        self.requests.append(list(messages))
        return self.replies.pop(0)


def tools(cards: Path, searcher: FakeSearcher, collapse: bool = True) -> CatalogueTools:
    s = SETTINGS.model_copy(update={"collapse_groups": collapse})
    return CatalogueTools(searcher, MANIFEST, cards, "c", {"equals": {"domain": "d"}}, s)


def test_search_collapses_members_into_their_group_and_discovers_ids(cards: Path) -> None:
    searcher = FakeSearcher(
        response("d/sat/sat-2015.csv.md", "d/codes.csv.md", "d/_groups/g-sat.md")
    )
    t = tools(cards, searcher)
    text = asyncio.run(t.search("density"))
    assert searcher.calls == [("density", {"equals": {"domain": "d"}}, 20)]
    assert "1. [group of 3 files] d/g-sat: sat-#.csv (d, group of 3 files) in d/sat" in text
    assert "matched member: d/sat/sat-2015.csv" in text
    assert "2. [file] id:d/codes.csv: d/codes.csv (10 bytes)" in text
    assert "3." not in text  # the group card itself was already shown
    assert t.discovered.ids == ["d/g-sat", "id:d/codes.csv"]
    plain = tools(cards, FakeSearcher(response("d/sat/sat-2015.csv.md")), collapse=False)
    asyncio.run(plain.search("density"))
    assert plain.discovered.ids == ["id:d/sat/sat-2015.csv"]


def test_read_card_is_bounded_lists_matching_members_and_refuses_undiscovered(
    cards: Path,
) -> None:
    t = tools(cards, FakeSearcher(response("d/_groups/g-sat.md")))
    assert "not returned by a search" in t.read_card("d/g-sat")
    asyncio.run(t.search("x"))
    card = t.read_card("d/g-sat", ("*", "*2016*"))
    assert "[card truncated at 200 characters]" in card
    assert "Members matching ['*', '*2016*']: 3 of 3" in card
    assert card.count("\n- d/sat/") == 2 and "- … 1 more" in card
    assert t.cards_read == 1


def test_choose_files_expands_filters_and_returns_problems_for_a_fix(cards: Path) -> None:
    script = Script(
        reply(("search_catalogue", {"query": "density"})),
        reply(("choose_files", {"selections": [{"id": "id:d/other.csv"}]})),
        reply(
            (
                "choose_files",
                {
                    "selections": [{"id": "d/g-sat", "members": ["*2015*", "*2016*"]}],
                    "reason": "density by year",
                },
            )
        ),
    )
    t = tools(cards, FakeSearcher(response("d/_groups/g-sat.md")))
    out = asyncio.run(
        run_discovery(script, t, user=user_prompt("q?", "d", 5), agent=AGENT, settings=SETTINGS)
    )
    assert out.stop_reason == "chosen" and out.steps == 3 and out.rejected_choices == 1
    assert out.chosen is not None and out.chosen.files == (SAT[1], SAT[2])
    assert out.chosen.items == (("d/g-sat", ("*2015*", "*2016*")),)
    assert out.searches == 1 and out.discovered == ("d/g-sat",)
    rejected = script.requests[2][-1]
    assert rejected.role == "tool" and "not returned by a search" in rejected.content
    # Append-only: each request extends the one before, with a fixed prefix.
    assert script.requests[0][0].content == SYSTEM_PROMPT
    assert script.requests[1][: len(script.requests[0])] == script.requests[0]


def test_budget_ends_without_files_and_warns_on_the_last_turn(cards: Path) -> None:
    s = SETTINGS.model_copy(update={"max_steps": 2, "budget_warning_steps": 1})
    script = Script(
        reply(("search_catalogue", {"query": "a"})), reply(("search_catalogue", {"query": "b"}))
    )
    t = tools(cards, FakeSearcher(response(), response("d/codes.csv.md")))
    out = asyncio.run(run_discovery(script, t, user="u", agent=AGENT, settings=s))
    assert out.stop_reason == "max_steps" and out.chosen is None
    assert script.requests[1][-1].content.endswith(LAST_TURN)
    assert "No results" in script.requests[1][-1].content
