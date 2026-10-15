"""A run's recorded steps and programs, arranged as a readable trace.

Each model request is a ledger step whose request holds the whole context so
far (context is append-only), so the conversation is the last request plus
its response, and the messages a request added are the slice between its
length and the next request's. Each slice starts with the context copy of
that request's reply, followed by tool results (or repair feedback).

Programs record their loop step: the agent loop counts one step per
``chat_with_repair`` call, which can span several requests when a tool call
is rejected and repaired, or a crashed runner is retried. Requests are
numbered back into loop steps with the loop's own acceptance check
(``check_response``), and each cell is placed under the tool call
(``call_index``) of the reply that ran it. A trace whose requests are not
append-only is shown request by request instead, and programs that cannot be
placed are listed rather than dropped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ds_research_agent.agent.tool_calls import check_response
from ds_research_agent.ledger import Program, Step
from ds_research_agent.models import ChatResult, ToolSpec

# Programs shown inline, under the call that ran them; the rest belong to the
# verification section (final_rerun) or the session (setup, session_end).
INLINE_KINDS = ("cell", "submit_check")
CODE_ARGS = ("code", "program")


@dataclass(frozen=True)
class CallView:
    index: int
    name: str
    args: dict[str, Any]  # without the code argument
    code: str | None
    programs: list[Program] = field(default_factory=list)


@dataclass(frozen=True)
class MessageView:
    role: str
    content: str
    thinking: str | None
    tool_name: str | None
    calls: list[CallView]


@dataclass
class StepView:
    step: Step
    ordinal: int  # 1-based among model requests
    loop_step: int  # the agent loop step this request belongs to
    accepted: bool  # the reply passed the loop's tool-call check
    started_s: float  # wall seconds into the run (sum of earlier requests)
    messages: list[MessageView]  # what this request added to the context
    searches: list[Step] = field(default_factory=list)  # discovery searches it led to
    programs: list[Program] = field(default_factory=list)  # placed, not under a call


@dataclass
class Trace:
    system: str | None
    # The tool definitions sent beside the messages on every request (Ollama's
    # ``tools`` field). The runner's chat template merges them into the prompt;
    # that rendered text is not recorded.
    tool_specs: list[dict[str, Any]]
    think: Any  # the request's thinking setting (False: off)
    options: dict[str, Any]
    preamble: list[MessageView]  # the task prompt, before the first request
    steps: list[StepView]
    append_only: bool
    session: list[Program]  # setup and session_end records
    final: list[Program]  # final_rerun records
    unplaced: list[Program]

    @property
    def tools(self) -> list[str]:
        return [str((t.get("function") or {}).get("name")) for t in self.tool_specs]


def message_view(m: dict[str, Any]) -> MessageView:
    calls = []
    for i, c in enumerate(m.get("tool_calls") or ()):
        fn = c.get("function", c)  # requests use Ollama's shape; responses, ToolCall's
        args = dict(fn.get("arguments") or {})
        code = next((str(args.pop(k)) for k in CODE_ARGS if k in args), None)
        calls.append(CallView(index=i, name=str(fn.get("name")), args=args, code=code))
    return MessageView(
        role=str(m.get("role")),
        content=str(m.get("content") or ""),
        thinking=m.get("thinking") or None,
        tool_name=m.get("tool_name"),
        calls=calls,
    )


def _accepted(step: Step) -> bool:
    if step.response is None:
        return False
    tools = step.request.get("tools") or []
    if not tools:  # no tool surface (codeagent, baselines): every reply is a step
        return True
    try:
        result = ChatResult.model_validate(step.response)
        specs = [ToolSpec.model_validate(t["function"]) for t in tools]
    except ValidationError, KeyError, TypeError:
        return False
    return not check_response(result, specs)


def _messages(step: Step) -> list[dict[str, Any]]:
    return list(step.request.get("messages") or [])


def _is_append_only(model: Sequence[Step]) -> bool:
    for a, b in zip(model, model[1:], strict=False):
        ma, mb = _messages(a), _messages(b)
        if mb[: len(ma)] != ma:
            return False
    return True


def build(steps: Sequence[Step], programs: Sequence[Program]) -> Trace:
    model = [s for s in steps if s.kind == "model"]
    req = model[0].request if model else {}
    specs = list(req.get("tools") or [])
    think, options = req.get("think"), dict(req.get("options") or {})
    by_step: dict[int, list[Program]] = defaultdict(list)
    session, final, unplaced = [], [], []
    for p in programs:
        if p.kind in INLINE_KINDS:
            by_step[p.step].append(p)
        elif p.kind == "final_rerun":
            final.append(p)
        else:
            session.append(p)
    if not model:
        orphans = [p for ps in by_step.values() for p in ps]
        return Trace(None, specs, think, options, [], [], True, session, final, orphans)

    append_only = _is_append_only(model)
    last = model[-1]
    transcript = _messages(last) + ([last.response["message"]] if last.response else [])
    first = _messages(model[0])
    system = next((str(m.get("content")) for m in first if m.get("role") == "system"), None)
    preamble = [message_view(m) for m in first if m.get("role") != "system"]

    views: list[StepView] = []
    loop_step, elapsed = 1, 0.0
    for i, s in enumerate(model):
        if append_only:
            start = len(_messages(s))
            end = len(_messages(model[i + 1])) if i + 1 < len(model) else len(transcript)
            added = [message_view(m) for m in transcript[start:end]]
        else:
            # Shown request by request: each later request in full (the first is
            # the preamble), then its reply.
            sent = [m for m in _messages(s) if m.get("role") != "system"] if i else []
            out = [s.response["message"]] if s.response else []
            added = [message_view(m) for m in sent + out]
        ok = _accepted(s)
        views.append(StepView(s, i + 1, loop_step, ok, round(elapsed, 1), added))
        elapsed += s.wall_s or 0.0
        if ok:
            loop_step += 1

    # Searches (discovery) follow the request whose reply asked for them.
    for s in steps:
        if s.kind == "search":
            owner = next((v for v in reversed(views) if v.step.seq < s.seq), None)
            if owner is not None:
                owner.searches.append(s)

    # Place each inline program under the accepted request of its loop step.
    for v in views:
        if not v.accepted:
            continue
        progs = by_step.pop(v.loop_step, [])
        reply = next((m for m in v.messages if m.role == "assistant"), None)
        calls = {c.index: c for c in reply.calls} if reply else {}
        for p in progs:
            c = calls.get(p.call_index)
            if c is not None and c.code == p.code:
                c.programs.append(p)
            else:  # codeagent replies carry code in their text, not a call
                v.programs.append(p)
    unplaced = [p for ps in by_step.values() for p in ps]
    return Trace(
        system, specs, think, options, preamble, views, append_only, session, final, unplaced
    )
