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
    ModelResponseError,
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


# done_reason when the client, not the server, ended the reply at a stop sequence.
CLIENT_STOP = "client_stop"


def first_stop(text: str, stop: Sequence[str]) -> int | None:
    """Where the earliest stop sequence starts in ``text``, if any."""
    found = [i for i in (text.find(s) for s in stop if s) if i >= 0]
    return min(found) if found else None


class OllamaModelClient:
    """With ``stop`` in the options, the reply is streamed and cut at the first
    stop sequence: Ollama 0.35.1's MLX runner ignores ``stop`` (measured
    2026-10-11; the GGUF runner honours it), so the client enforces it. The
    cut reply excludes the sequence, as a server stop does, and the stream is
    closed so generation ends there too. It then has no server counters
    (``Usage`` is empty) and ``done_reason`` is ``CLIENT_STOP``."""

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._client = ollama.AsyncClient(host=settings.host, timeout=settings.request_timeout_s)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> ChatResult:
        s = self._settings
        stop = [str(x) for x in s.options.get("stop") or ()]
        kwargs: dict[str, Any] = {
            "model": s.name,
            "messages": [to_ollama_message(m) for m in messages],
            "tools": [to_ollama_tool(t) for t in tools] or None,
            "think": s.think,
            "options": s.options or None,
            "keep_alive": s.keep_alive,
        }
        start = time.monotonic()
        try:
            if stop and not tools:
                return await self._until_stop(kwargs, stop, start)
            resp = await self._client.chat(**kwargs)
        except ollama.ResponseError as e:
            raise ModelResponseError(e.error, e.status_code) from e
        return from_ollama_response(resp, time.monotonic() - start)

    async def _until_stop(
        self, kwargs: dict[str, Any], stop: Sequence[str], start: float
    ) -> ChatResult:
        stream = await self._client.chat(**kwargs, stream=True)
        content, thinking = "", ""
        last: ollama.ChatResponse | None = None
        try:
            async for part in stream:
                last = part
                content += part.message.content or ""
                thinking += part.message.thinking or ""
                cut = first_stop(content, stop)
                if cut is not None:
                    message = ChatMessage(
                        role="assistant", content=content[:cut], thinking=thinking or None
                    )
                    return ChatResult(
                        message=message,
                        usage=Usage(),
                        done_reason=CLIENT_STOP,
                        model=part.model,
                        wall_s=time.monotonic() - start,
                    )
        finally:
            # An async generator at runtime (typed as an iterator): closing it
            # closes the response, so the server stops generating.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        if last is None:
            raise ModelResponseError("empty stream")
        final = last.model_copy(
            update={
                "message": last.message.model_copy(
                    update={"content": content, "thinking": thinking or None}
                )
            }
        )
        return from_ollama_response(final, time.monotonic() - start)
