from collections.abc import Sequence

from ds_research_agent.agent import chat_with_repair, check_calls, check_response
from ds_research_agent.models import (
    ChatMessage,
    ChatResult,
    ModelResponseError,
    ToolCall,
    ToolSpec,
    Usage,
)

SEARCH = ToolSpec(
    name="search",
    description="Search.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


def reply(*calls: ToolCall, content: str = "") -> ChatResult:
    return ChatResult(
        message=ChatMessage(role="assistant", content=content, tool_calls=calls),
        usage=Usage(),
        wall_s=0.0,
    )


class Scripted:
    """Returns (or raises) queued responses and records every request."""

    def __init__(self, *responses: ChatResult | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        self.requests.append(list(messages))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_valid_call_has_no_problems() -> None:
    assert (
        check_calls([ToolCall(name="search", arguments={"query": "q", "top_k": 3})], [SEARCH]) == []
    )


def test_argument_faults_are_classified() -> None:
    [p] = check_calls([ToolCall(name="search", arguments={"top_k": "three", "extra": 1})], [SEARCH])
    assert p.kind == "invalid_arguments"
    assert sorted(p.faults) == ["missing_required", "unexpected_argument", "wrong_type"]
    assert "top_k" in p.detail


def test_unknown_tool() -> None:
    [p] = check_calls([ToolCall(name="plot", arguments={})], [SEARCH])
    assert (p.kind, p.call_index) == ("unknown_tool", 0)
    assert "search" in p.detail


def test_markup_left_in_content_is_unparsed() -> None:
    [p] = check_response(reply(content="<tool_call>\n<function=search>"), [SEARCH])
    assert p.kind == "unparsed_markup"


def test_plain_answer_is_accepted() -> None:
    assert check_response(reply(content="The answer is 4."), [SEARCH]) == []


async def test_invalid_call_is_repaired_append_only() -> None:
    bad = reply(ToolCall(name="search", arguments={"query": "q", "top_k": "three"}))
    good = reply(ToolCall(name="search", arguments={"query": "q", "top_k": 3}))
    client = Scripted(bad, good)
    base = [ChatMessage(role="user", content="find q")]
    out = await chat_with_repair(client, base, [SEARCH], max_repairs=2)

    assert out.result is good
    assert [r.attempt for r in out.repairs] == [0]
    assert out.repairs[0].problems[0].faults == ("wrong_type",)
    # The retry request extends the first one: nothing earlier is rewritten.
    first, second = client.requests
    assert second[: len(first)] == first
    assert second[len(first)] == bad.message
    assert second[len(first) + 1].role == "tool"
    assert "top_k" in second[len(first) + 1].content
    assert out.appended == (*second[len(first) :], good.message)


async def test_parse_error_appends_only_feedback() -> None:
    good = reply(ToolCall(name="search", arguments={"query": "q"}))
    client = Scripted(ModelResponseError("XML syntax error", 500), good)
    out = await chat_with_repair(client, [ChatMessage(role="user", content="u")], [SEARCH], 1)

    assert out.result is good
    assert out.repairs[0].problems[0].kind == "parse_error"
    assert out.repairs[0].calls is None
    assert [m.role for m in client.requests[1]] == ["user", "user"]
    assert "XML syntax error" in client.requests[1][1].content


async def test_valid_sibling_call_is_reported_not_run() -> None:
    mixed = reply(
        ToolCall(name="search", arguments={"query": "a"}),
        ToolCall(name="search", arguments={}),
    )
    client = Scripted(mixed, reply(content="done"))
    out = await chat_with_repair(client, [ChatMessage(role="user", content="u")], [SEARCH], 1)
    tool_msgs = [m for m in out.appended if m.role == "tool"]
    assert tool_msgs[0].content.startswith("Not run")
    assert tool_msgs[1].content.startswith("Error")
    assert out.repairs[0].problems[0].call_index == 1


async def test_repairs_are_bounded() -> None:
    bad = reply(ToolCall(name="plot", arguments={}))
    client = Scripted(bad, bad, bad)
    out = await chat_with_repair(client, [ChatMessage(role="user", content="u")], [SEARCH], 2)
    assert out.exhausted
    assert [r.attempt for r in out.repairs] == [0, 1, 2]
    assert len(client.requests) == 3


async def test_zero_repairs_disables_retry() -> None:
    client = Scripted(reply(ToolCall(name="plot", arguments={})))
    out = await chat_with_repair(client, [ChatMessage(role="user", content="u")], [SEARCH], 0)
    assert out.exhausted and out.appended == ()
