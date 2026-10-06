import ollama

from ds_research_agent.models import ChatMessage, ToolCall, ToolSpec
from ds_research_agent.models.ollama_client import (
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
