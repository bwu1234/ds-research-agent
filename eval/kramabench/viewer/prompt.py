"""The prompt text a recorded request became, approximately.

Requests go to Ollama's ``/api/chat``, whose compiled-in renderer (``qwen3.8``
for qwen3.8:27b-mlx, per the model's image config) merges the messages and
the ``tools`` field into the prompt the model reads. Ollama does not return
that text, so the ledger does not have it. This renders a recorded request
through the model's own chat template, the ``chat_template`` of its
``tokenizer_config.json`` layer, as transformers' ``apply_chat_template``
does, with three inputs prepared the way Ollama 0.40.4's renderer
(``model/renderers/qwen35.go``) prepares them:

- each tool is re-encoded from Ollama's ``api.Tool`` structs: keys the
  structs lack (``additionalProperties``) are dropped, keys come in struct
  order, a one-element ``type`` list is a string, and Go's JSON escapes
  ``<``, ``>``, and ``&``;
- a non-string tool-call argument is Go's ``%v`` (numbers, booleans) or
  compact Go JSON (lists, objects), not the template's spaced JSON;
- ``think`` maps to the template's variables as the renderer reads it
  (unset: xhigh; ``true``: medium; ``false``: thinking off).

It stays an approximation: the ledger stores requests as canonical JSON, so
object keys (tool properties, call arguments) are sorted rather than in the
order sent, and Ollama versions before 0.40 rendered differently. Pages label
it accordingly.

Templates are vendored per model (``chat_templates/``) and keyed by the
batch's model name; a model without one gets no rendering rather than another
model's format.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_DIR = Path(__file__).parent / "chat_templates"
# Model name -> template extracted from that model's tokenizer_config.json
# layer (sha256 c3cf9e34... for qwen3.8:27b-mlx, extracted 2026-10-15).
TEMPLATES = {"qwen3.8:27b-mlx": "qwen3.8_27b_mlx.jinja"}
RENDERER = "Ollama 0.40.4 qwen3.8"  # the renderer the preparation follows


class NoTemplate(LookupError):
    pass


@dataclass(frozen=True)
class Rendered:
    text: str
    template: str  # file name
    variables: dict[str, Any]  # the template variables the think setting became

    @property
    def system_turn(self) -> str:
        """The first turn (system prompt with the tool definitions merged in)."""
        end = "<|im_end|>\n"
        i = self.text.find(end)
        return self.text if i < 0 else self.text[: i + len(end)]


# ---- Go's encodings ---------------------------------------------------------


class _GoJSON(str):
    """Text already encoded; the template's tojson emits it unchanged."""

    __slots__ = ()


def _go_float(v: float, *, json_style: bool) -> str:
    """Go's shortest float formatting: encoding/json's (``json_style``) or fmt's %v."""
    if math.isnan(v) or math.isinf(v):
        return "null" if json_style else ("NaN" if math.isnan(v) else ("+Inf" if v > 0 else "-Inf"))
    if v == 0:
        return "0"
    d = Decimal(repr(v))
    exp = d.adjusted()
    if json_style:  # encoding/json floatEncoder
        use_e = abs(v) < 1e-6 or abs(v) >= 1e21
    else:  # strconv 'g' with shortest precision: eprec 6
        use_e = exp < -4 or exp >= 6
    if not use_e:
        s = format(d, "f")
        return s.rstrip("0").rstrip(".") if "." in s else s
    digits = d.as_tuple().digits
    mant = str(digits[0]) + (
        "." + "".join(map(str, digits[1:])).rstrip("0") if len(digits) > 1 else ""
    )
    mant = mant.rstrip(".")
    sign = "-" if v < 0 else ""
    e = f"{exp:+03d}" if not json_style else f"{exp:+d}"
    return f"{sign}{mant}e{e}"


def _go_string(s: str) -> str:
    out = json.dumps(s, ensure_ascii=False)
    for a, b in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        out = out.replace(a, b)
    return out.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _go_json(v: Any) -> str:
    """Compact Go encoding/json: maps in their given order (struct order for the
    tool structs; the ledger's sorted order otherwise), HTML characters escaped."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _go_float(v, json_style=True)
    if isinstance(v, str):
        return _go_string(v)
    if isinstance(v, Mapping):
        return "{" + ",".join(f"{_go_string(str(k))}:{_go_json(x)}" for k, x in v.items()) + "}"
    if isinstance(v, Sequence):
        return "[" + ",".join(_go_json(x) for x in v) + "]"
    return _go_string(str(v))


def _add_spaces(b: str) -> str:
    """Ollama's AddJSONSpaces: a space after ':' and ',' outside strings."""
    out, in_str, esc = [], False, False
    for c in b:
        out.append(c)
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in ":,":
            out.append(" ")
    return "".join(out)


def _property(p: Mapping[str, Any]) -> dict[str, Any]:
    """api.ToolProperty field order, empty fields omitted."""
    out: dict[str, Any] = {}
    if p.get("anyOf"):
        out["anyOf"] = [_property(x) for x in p["anyOf"]]
    t = p.get("type")
    types = [t] if isinstance(t, str) else list(t or [])
    if types:
        out["type"] = types[0] if len(types) == 1 else types
    if p.get("items") is not None:
        out["items"] = p["items"]
    if p.get("description"):
        out["description"] = p["description"]
    if p.get("enum"):
        out["enum"] = p["enum"]
    if p.get("properties") is not None:
        out["properties"] = {k: _property(x) for k, x in p["properties"].items()}
    if p.get("required"):
        out["required"] = p["required"]
    return out


def ollama_tool(t: Mapping[str, Any]) -> str:
    """A tool as Ollama's renderer writes it: api.Tool re-encoded, spaced."""
    fn = t.get("function") or {}
    p = fn.get("parameters") or {}
    params: dict[str, Any] = {"type": p.get("type", "")}
    for k in ("$defs", "items"):
        if p.get(k) is not None:
            params[k] = p[k]
    if p.get("required"):
        params["required"] = p["required"]
    props = p.get("properties")
    params["properties"] = None if props is None else {k: _property(x) for k, x in props.items()}
    function: dict[str, Any] = {"name": fn.get("name", "")}
    if fn.get("description"):
        function["description"] = fn["description"]
    function["parameters"] = params
    tool: dict[str, Any] = {"type": t.get("type", "")}
    if t.get("items") is not None:
        tool["items"] = t["items"]
    tool["function"] = function
    return _add_spaces(_go_json(tool))


def tool_call_argument(v: Any) -> str:
    """Ollama's formatToolCallArgument: strings as is, lists and objects as
    compact Go JSON, everything else Go's %v (numbers decode as float64)."""
    if v is None:
        return "null"
    if isinstance(v, str):
        return v
    if isinstance(v, Mapping | list | tuple):
        return _go_json(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return _go_float(float(v), json_style=False)
    return str(v)


# ---- rendering --------------------------------------------------------------


def _raise(msg: str) -> None:
    raise ValueError(msg)


def _tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:  # fmt: skip
    if isinstance(x, _GoJSON):
        return str(x)
    # transformers' tojson: no ASCII escaping, keys in their given order.
    return json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


@cache
def _template(name: str) -> Any:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise
    env.filters["tojson"] = _tojson
    return env.from_string((TEMPLATE_DIR / name).read_text(encoding="utf-8"))


def template_variables(think: Any) -> dict[str, Any]:
    """Ollama's think value as the template's variables, resolved the way the
    qwen3.8 renderer resolves it (qwen38ReasoningInstructions)."""
    if think is False:
        return {"enable_thinking": False}
    effort = {None: "xhigh", True: "medium", "low": "low", "medium": "medium"}.get(think, "xhigh")
    return {"enable_thinking": True, "reasoning_effort": effort}


def _message(m: Mapping[str, Any]) -> dict[str, Any]:
    """Ollama's message shape to the template's: reasoning as reasoning_content,
    tool-call arguments formatted as the renderer formats them."""
    out = {k: v for k, v in m.items() if k not in ("thinking", "tool_name", "tool_calls")}
    if m.get("thinking"):
        out["reasoning_content"] = m["thinking"]
    calls = []
    for c in m.get("tool_calls") or ():
        fn = c.get("function", c)
        args = {k: tool_call_argument(v) for k, v in (fn.get("arguments") or {}).items()}
        calls.append({"function": {"name": fn.get("name"), "arguments": args}})
    if calls:
        out["tool_calls"] = calls
    return out


def render(model: str, request: Mapping[str, Any]) -> Rendered:
    name = TEMPLATES.get(model)
    if name is None:
        raise NoTemplate(f"no chat template vendored for {model!r}")
    variables = template_variables(request.get("think"))
    messages: Sequence[Mapping[str, Any]] = request.get("messages") or []
    tools = [_GoJSON(ollama_tool(t)) for t in request.get("tools") or []]
    text = _template(name).render(
        messages=[_message(m) for m in messages],
        tools=tools or None,
        add_generation_prompt=True,
        **variables,
    )
    return Rendered(text=text, template=name, variables=variables)
