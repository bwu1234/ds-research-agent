"""The viewer's approximate prompt rendering, offline: the vendored template,
the think mapping, and the Ollama 0.40.4 encodings it mirrors."""

from __future__ import annotations

import hashlib

import pytest

from eval.kramabench.viewer import prompt

MODEL = "qwen3.8:27b-mlx"
TOOL = {
    "type": "function",
    "function": {
        "name": "read_card",
        "description": "Read a <card> & its members.",
        "parameters": {
            "additionalProperties": False,
            "properties": {
                "id": {"description": "An ID.", "type": "string"},
                "members": {"items": {"type": "string"}, "type": ["array"]},
            },
            "required": ["id"],
            "type": "object",
        },
    },
}


def test_vendored_template_is_the_extracted_one() -> None:
    data = (prompt.TEMPLATE_DIR / prompt.TEMPLATES[MODEL]).read_bytes()
    assert hashlib.sha256(data).hexdigest().startswith("c3cf9e34abf4f9e3")


def test_think_resolves_like_the_qwen38_renderer() -> None:
    assert prompt.template_variables(False) == {"enable_thinking": False}
    effort = {v: prompt.template_variables(v)["reasoning_effort"] for v in (None, True, "low")}
    assert effort == {None: "xhigh", True: "medium", "low": "low"}
    assert prompt.template_variables("high")["reasoning_effort"] == "xhigh"


def test_tools_are_reencoded_like_ollamas_structs() -> None:
    # additionalProperties dropped, struct key order, a one-type list as a
    # string, HTML characters escaped, a space after ':' and ','.
    assert prompt.ollama_tool(TOOL) == (
        '{"type": "function", "function": {"name": "read_card", "description": '
        '"Read a \\u003ccard\\u003e \\u0026 its members.", "parameters": {"type": "object", '
        '"required": ["id"], "properties": {"id": {"type": "string", "description": "An ID."}, '
        '"members": {"type": "array", "items": {"type": "string"}}}}}}'
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("as is", "as is"),
        (3, "3"),
        (3.0, "3"),
        (0.25, "0.25"),
        (1234567.0, "1.234567e+06"),
        (0.00001, "1e-05"),
        (True, "true"),
        (None, "null"),
        (["a", 1.0, {"k": "<x>"}], '["a",1,{"k":"\\u003cx\\u003e"}]'),
    ],
)
def test_tool_call_arguments_use_gos_formatting(value: object, text: str) -> None:
    assert prompt.tool_call_argument(value) == text


def test_render_merges_tools_into_the_system_turn() -> None:
    request = {
        "think": False,
        "tools": [TOOL],
        "messages": [
            {"role": "system", "content": "Be careful."},
            {"role": "user", "content": "Question: q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_card", "arguments": {"id": "x", "members": ["*"]}}}
                ],
            },
            {"role": "tool", "tool_name": "read_card", "content": "card"},
        ],
    }
    r = prompt.render(MODEL, request)
    sys_turn = r.system_turn
    assert (
        sys_turn.startswith("<|im_start|>system\n# Tools\n") and "Reasoning effort" not in sys_turn
    )
    assert prompt.ollama_tool(TOOL) in sys_turn and sys_turn.endswith("Be careful.<|im_end|>\n")
    assert '<parameter=members>\n["*"]\n</parameter>' in r.text
    assert "<tool_response>\ncard\n</tool_response><|im_end|>\n" in r.text
    assert r.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    low = prompt.render(MODEL, request | {"think": "low"})
    assert "Reasoning effort is set to low." in low.system_turn
    assert low.text.endswith("<|im_start|>assistant\n<think>\n")


def test_unknown_model_gets_no_rendering() -> None:
    with pytest.raises(prompt.NoTemplate):
        prompt.render("other:model", {"messages": [{"role": "user", "content": "q"}]})
