"""One tool-calling round trip through the configured local model."""

import pytest

from ds_research_agent.config import Settings
from ds_research_agent.models import ChatMessage, ToolSpec
from ds_research_agent.models.ollama_client import OllamaModelClient

pytestmark = pytest.mark.live

ADD = ToolSpec(
    name="add",
    description="Add two integers and return the sum.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)


async def test_tool_call_round_trip(settings: Settings) -> None:
    client = OllamaModelClient(settings.model)
    messages = [
        ChatMessage(role="system", content="Use the add tool for arithmetic."),
        ChatMessage(role="user", content="What is 1234 + 8766? Use the tool."),
    ]
    first = await client.chat(messages, [ADD])
    assert len(first.message.tool_calls) == 1, first.message
    call = first.message.tool_calls[0]
    assert call.name == "add"
    assert {int(call.arguments["a"]), int(call.arguments["b"])} == {1234, 8766}

    messages += [
        first.message,
        ChatMessage(role="tool", tool_name="add", content=str(10000)),
    ]
    second = await client.chat(messages, [ADD])
    assert not second.message.tool_calls
    assert "10000" in second.message.content.replace(",", "")
    for i, r in enumerate((first, second), 1):
        u = r.usage
        print(
            f"\nturn {i}: wall {r.wall_s:.1f}s, prompt_eval {u.prompt_eval_tokens}, "
            f"output {u.output_tokens}, thinking chars {len(r.message.thinking or '')}"
        )
