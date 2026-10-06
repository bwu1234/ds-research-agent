"""Model client interface, independent of any provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_Frozen):
    name: str
    description: str
    # JSON Schema for the arguments object.
    parameters: dict[str, Any]


class ToolCall(_Frozen):
    name: str
    arguments: dict[str, Any]


class ChatMessage(_Frozen):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    thinking: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    # Set on role="tool": which tool produced this result.
    tool_name: str | None = None


class Usage(_Frozen):
    """Per-request counters exactly as the provider reports them.

    With Ollama 0.35.1, ``prompt_eval_tokens`` is the whole prompt even when
    a cached prefix was reused (measured in D0); only ``prompt_eval_ns``
    reflects how much was actually evaluated.
    """

    prompt_eval_tokens: int | None = None
    output_tokens: int | None = None
    total_ns: int | None = None
    load_ns: int | None = None
    prompt_eval_ns: int | None = None
    eval_ns: int | None = None


class ChatResult(_Frozen):
    message: ChatMessage
    usage: Usage
    done_reason: str | None = None
    model: str | None = None
    # Wall-clock time measured by the client, including transport.
    wall_s: float = Field(ge=0)


class ModelResponseError(Exception):
    """The provider failed a request instead of returning a message.

    Ollama 0.35.1 does this when it cannot parse the model's tool-call
    markup (the qwen3.5 parser returns an error and the request is
    cancelled), so there is no assistant message to repair.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class ModelClient(Protocol):
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> ChatResult: ...
