"""Ollama implementation of the model client."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import ollama

from ds_research_agent.config import ModelSettings
from ds_research_agent.models.base import (
    ChatMessage,
    ChatResult,
    ToolCall,
    ToolSpec,
    Usage,
)


def to_ollama_message(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.thinking is not None:
        out["thinking"] = m.thinking
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    if m.tool_name is not None:
        out["tool_name"] = m.tool_name
    return out


def to_ollama_tool(t: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
    }


def from_ollama_response(resp: ollama.ChatResponse, wall_s: float) -> ChatResult:
    msg = resp.message
    calls = tuple(
        ToolCall(name=c.function.name, arguments=dict(c.function.arguments))
        for c in (msg.tool_calls or ())
    )
    return ChatResult(
        message=ChatMessage(
            role="assistant",
            content=msg.content or "",
            thinking=msg.thinking or None,
            tool_calls=calls,
        ),
        usage=Usage(
            prompt_eval_tokens=resp.prompt_eval_count,
            output_tokens=resp.eval_count,
            total_ns=resp.total_duration,
            load_ns=resp.load_duration,
            prompt_eval_ns=resp.prompt_eval_duration,
            eval_ns=resp.eval_duration,
        ),
        done_reason=resp.done_reason,
        model=resp.model,
        wall_s=wall_s,
    )


class OllamaModelClient:
    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._client = ollama.AsyncClient(host=settings.host, timeout=settings.request_timeout_s)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> ChatResult:
        s = self._settings
        start = time.monotonic()
        resp = await self._client.chat(
            model=s.name,
            messages=[to_ollama_message(m) for m in messages],
            tools=[to_ollama_tool(t) for t in tools] or None,
            think=s.think,
            options=s.options or None,
            keep_alive=s.keep_alive,
        )
        return from_ollama_response(resp, time.monotonic() - start)
