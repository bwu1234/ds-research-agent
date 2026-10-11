from typing import Any

import ollama

from ds_research_agent.config import ModelSettings
from ds_research_agent.models import ChatMessage, ToolCall, ToolSpec
from ds_research_agent.models.ollama_client import (
    CLIENT_STOP,
    OllamaModelClient,
    first_stop,
    from_ollama_response,
    to_ollama_message,
    to_ollama_tool,
)


def test_assistant_tool_call_round_trips_to_wire() -> None:
    m = ChatMessage(
        role="assistant",
        thinking="t",
        tool_calls=(ToolCall(name="add", arguments={"a": 1}),),
    )
    assert to_ollama_message(m) == {
        "role": "assistant",
        "content": "",
        "thinking": "t",
        "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1}}}],
    }


def test_tool_result_message_carries_tool_name() -> None:
    m = ChatMessage(role="tool", content="3", tool_name="add")
    assert to_ollama_message(m)["tool_name"] == "add"


def test_tool_spec_wire_shape() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    t = ToolSpec(name="add", description="Add.", parameters=schema)
    assert to_ollama_tool(t)["function"]["parameters"] == schema


def test_response_maps_calls_and_usage() -> None:
    resp = ollama.ChatResponse(
        model="m",
        done=True,
        done_reason="stop",
        prompt_eval_count=120,
        eval_count=30,
        total_duration=5,
        message=ollama.Message(
            role="assistant",
            content="",
            thinking="why",
            tool_calls=[
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(name="add", arguments={"a": 1})
                )
            ],
        ),
    )
    r = from_ollama_response(resp, wall_s=0.5)
    assert r.message.tool_calls == (ToolCall(name="add", arguments={"a": 1}),)
    assert r.message.thinking == "why"
    assert r.usage.prompt_eval_tokens == 120
    assert r.usage.output_tokens == 30
    assert r.done_reason == "stop"


class _StreamingFake:
    """``AsyncClient.chat``: the reply in ``chunks`` when streamed, else one response."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks, self.calls, self.closed = chunks, [], False  # type: ignore[var-annotated]

    async def chat(self, **kw: Any) -> Any:
        self.calls.append(kw)
        msg = lambda c: ollama.Message(role="assistant", content=c)  # noqa: E731
        if not kw.get("stream"):
            return ollama.ChatResponse(model="m", done=True, message=msg("".join(self.chunks)))
        fake = self

        async def gen() -> Any:
            try:
                for i, c in enumerate(fake.chunks):
                    last = i == len(fake.chunks) - 1
                    yield ollama.ChatResponse(
                        model="m", done=last, done_reason="stop" if last else None,
                        eval_count=7 if last else None, message=msg(c),
                    )  # fmt: skip
            finally:
                fake.closed = True

        return gen()


def _client(chunks: list[str], options: dict[str, Any]) -> tuple[OllamaModelClient, Any]:
    c = OllamaModelClient(
        ModelSettings(name="m", think=False, request_timeout_s=5, options=options)
    )
    fake = _StreamingFake(chunks)
    c._client = fake  # type: ignore[assignment]
    return c, fake


async def test_stop_sequences_are_enforced_by_the_client() -> None:
    """Ollama 0.35.1's MLX runner ignores ``stop``; the client cuts the reply
    at the first stop sequence, even one split across chunks, and closes the stream."""
    stop = ["Observation:", "</code>"]
    c, fake = _client(["<code>\nprint(1)\n</co", "de>\nObserv", "ation: 1"], {"stop": stop})
    r = await c.chat([ChatMessage(role="user", content="q")])
    assert r.message.content == "<code>\nprint(1)\n"
    assert r.done_reason == CLIENT_STOP and r.usage.output_tokens is None
    assert fake.calls[0]["stream"] is True and fake.calls[0]["options"]["stop"] == stop
    assert fake.closed


async def test_stream_without_a_stop_sequence_keeps_the_server_counters() -> None:
    c, _ = _client(["Thought: done", "."], {"stop": ["</code>"]})
    r = await c.chat([ChatMessage(role="user", content="q")])
    assert r.message.content == "Thought: done." and r.done_reason == "stop"
    assert r.usage.output_tokens == 7


async def test_no_stop_option_is_one_unstreamed_request() -> None:
    c, fake = _client(["a</code>b"], {"seed": 0})
    r = await c.chat([ChatMessage(role="user", content="q")])
    assert r.message.content == "a</code>b" and "stream" not in fake.calls[0]


def test_first_stop() -> None:
    assert first_stop("ab</code>Observation:", ["Observation:", "</code>"]) == 2
    assert first_stop("abc", ["x"]) is None and first_stop("abc", [""]) is None
