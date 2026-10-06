"""Tool-call validation and the bounded repair-and-retry policy.

A model step is accepted when it either makes no tool call or every call
names an offered tool with arguments that validate against its JSON Schema.
Otherwise the step is repaired by appending feedback and asking again, up to
``max_repairs`` times. Repairs only ever append to the conversation, so the
cached prompt prefix survives (see the prefix-cache measurements in
``docs/implementation-plan.md``).

Problems, as Ollama 0.35.1 surfaces them for the qwen3.5 parser:

- ``parse_error``: the server could not parse the model's tool-call markup
  and failed the request, so no assistant message exists. The repair
  appends only a user message describing the error.
- ``unparsed_markup``: the reply contains tool-call markup but no parsed
  call (an unclosed ``<tool_call>`` is returned as content).
- ``unknown_tool``: a call names a tool that was not offered.
- ``invalid_arguments``: the arguments fail the tool's schema. Ollama has
  already coerced each value toward its declared type, so a wrong type here
  means coercion failed (for example ``"three"`` for an integer).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict

from ds_research_agent.models import (
    ChatMessage,
    ChatResult,
    ModelClient,
    ModelResponseError,
    ToolCall,
    ToolSpec,
)

ProblemKind = Literal["parse_error", "unparsed_markup", "unknown_tool", "invalid_arguments"]
# Sub-kind for invalid_arguments, from the failing JSON Schema keyword.
ArgumentFault = Literal["wrong_type", "missing_required", "unexpected_argument", "other"]

# Qwen tool-call markup that should never survive parsing into content.
TOOL_MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<parameter=")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CallProblem(_Frozen):
    kind: ProblemKind
    detail: str
    tool: str | None = None
    # Position of the offending call in the message, when there is one.
    call_index: int | None = None
    # Only for invalid_arguments.
    faults: tuple[ArgumentFault, ...] = ()


class RepairRecord(_Frozen):
    """One rejected model response. Ledger rows for a step's repairs."""

    attempt: int  # 0 = the step's first response, n = after the n-th repair
    problems: tuple[CallProblem, ...]
    # The rejected calls exactly as parsed, or None for a parse_error.
    calls: tuple[ToolCall, ...] | None
    wall_s: float


class StepOutcome(_Frozen):
    # The accepted response, or None when repairs ran out.
    result: ChatResult | None
    # Messages to append to the conversation, in order. Ends with the
    # accepted assistant message when ``result`` is set.
    appended: tuple[ChatMessage, ...]
    repairs: tuple[RepairRecord, ...]
    wall_s: float

    @property
    def exhausted(self) -> bool:
        return self.result is None


_FAULT_BY_KEYWORD: dict[str, ArgumentFault] = {
    "type": "wrong_type",
    "required": "missing_required",
    "additionalProperties": "unexpected_argument",
}


def _validators(tools: Sequence[ToolSpec]) -> dict[str, Draft202012Validator]:
    out = {}
    for t in tools:
        Draft202012Validator.check_schema(t.parameters)
        out[t.name] = Draft202012Validator(t.parameters)
    return out


def check_calls(calls: Sequence[ToolCall], tools: Sequence[ToolSpec]) -> list[CallProblem]:
    """Problems with each call; empty when all calls are acceptable."""
    validators = _validators(tools)
    problems = []
    for i, c in enumerate(calls):
        v = validators.get(c.name)
        if v is None:
            offered = ", ".join(sorted(validators))
            problems.append(
                CallProblem(
                    kind="unknown_tool",
                    tool=c.name,
                    call_index=i,
                    detail=f"There is no tool named {c.name!r}. Available tools: {offered}.",
                )
            )
            continue
        errors = sorted(v.iter_errors(c.arguments), key=lambda e: list(e.absolute_path))
        if errors:
            problems.append(
                CallProblem(
                    kind="invalid_arguments",
                    tool=c.name,
                    call_index=i,
                    detail="; ".join(_describe(e.absolute_path, e.message) for e in errors),
                    faults=tuple(_FAULT_BY_KEYWORD.get(str(e.validator), "other") for e in errors),
                )
            )
    return problems


def _describe(path: Any, message: str) -> str:
    where = "/".join(str(p) for p in path)
    return f"{where}: {message}" if where else message


def check_response(result: ChatResult, tools: Sequence[ToolSpec]) -> list[CallProblem]:
    m = result.message
    if not m.tool_calls and any(tag in m.content for tag in TOOL_MARKUP):
        return [
            CallProblem(
                kind="unparsed_markup",
                detail="The reply contains tool-call markup that could not be parsed as a call.",
            )
        ]
    return check_calls(m.tool_calls, tools)


# Repair feedback is fixed text plus the specific problem, so repaired
# conversations stay byte-stable across runs.
def _feedback(problems: Sequence[CallProblem]) -> str:
    lines = "\n".join(f"- {p.detail}" for p in problems)
    return (
        f"Your previous tool call was rejected and nothing was run:\n{lines}\n"
        "Reply again with a corrected tool call."
    )


def repair_messages(
    response: ChatMessage | None, problems: Sequence[CallProblem]
) -> list[ChatMessage]:
    """Messages appended after a rejected response.

    Every call in a rejected message is answered with a tool result so the
    transcript stays well-formed; calls that were themselves acceptable are
    reported as not run.
    """
    if response is None:  # parse_error: the failed output never reached us
        return [ChatMessage(role="user", content=_feedback(problems))]
    out = [response]
    if not response.tool_calls:  # unparsed_markup
        out.append(ChatMessage(role="user", content=_feedback(problems)))
        return out
    by_index = {p.call_index: p for p in problems}
    for i, c in enumerate(response.tool_calls):
        p = by_index.get(i)
        if p is None:
            content = "Not run: another tool call in the same message was rejected."
        elif p.kind == "unknown_tool":
            content = f"Error: {p.detail} Nothing was run."
        else:
            content = (
                f"Error: {p.detail} Nothing was run. Call {c.name} again with corrected arguments."
            )
        out.append(ChatMessage(role="tool", tool_name=c.name, content=content))
    return out


async def chat_with_repair(
    client: ModelClient,
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolSpec],
    max_repairs: int,
) -> StepOutcome:
    """Run one model step, repairing rejected tool calls up to ``max_repairs`` times."""
    start = time.monotonic()
    appended: list[ChatMessage] = []
    repairs: list[RepairRecord] = []
    for attempt in range(max_repairs + 1):
        t0 = time.monotonic()
        try:
            result = await client.chat([*messages, *appended], tools)
        except ModelResponseError as e:
            problems = [
                CallProblem(kind="parse_error", detail=f"Tool call not parsed: {e.message}")
            ]
            repairs.append(
                RepairRecord(
                    attempt=attempt,
                    problems=tuple(problems),
                    calls=None,
                    wall_s=time.monotonic() - t0,
                )
            )
            if attempt < max_repairs:
                appended += repair_messages(None, problems)
            continue
        problems = check_response(result, tools)
        if not problems:
            appended.append(result.message)
            return StepOutcome(
                result=result,
                appended=tuple(appended),
                repairs=tuple(repairs),
                wall_s=time.monotonic() - start,
            )
        repairs.append(
            RepairRecord(
                attempt=attempt,
                problems=tuple(problems),
                calls=result.message.tool_calls,
                wall_s=result.wall_s,
            )
        )
        if attempt < max_repairs:
            appended += repair_messages(result.message, problems)
    return StepOutcome(
        result=None,
        appended=tuple(appended),
        repairs=tuple(repairs),
        wall_s=time.monotonic() - start,
    )
