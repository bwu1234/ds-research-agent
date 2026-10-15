"""A small HTML builder that escapes by default.

Ledger text (model output, tool results, data file contents) is untrusted,
so every string is escaped unless it is already ``Html`` built here. Pages
are also served with a Content-Security-Policy that forbids inline scripts,
so an escaping mistake does not become script execution.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlencode

type Child = str | int | float | Html | None | Iterable[Child]


class Html(str):
    """Markup that is already safe to emit."""

    __slots__ = ()


def esc(v: object) -> Html:
    return v if isinstance(v, Html) else Html(html.escape(str(v), quote=True))


def _children(cs: Iterable[Child]) -> Iterable[Html]:
    for c in cs:
        if c is None:
            continue
        if isinstance(c, str | int | float):
            yield esc(c)
        else:
            yield from _children(c)


def join(*children: Child) -> Html:
    return Html("".join(_children(children)))


_VOID = {"br", "hr", "input", "meta", "link", "col"}


def tag(name: str, /, *children: Child, **attrs: Any) -> Html:
    """``<name attrs>children</name>``. Attribute names: a trailing underscore
    is dropped (``class_``), other underscores become hyphens (``data_key``).
    False or None attributes are left out; True ones are written bare."""
    parts = [name]
    for k, v in attrs.items():
        if v is None or v is False:
            continue
        k = k.removesuffix("_").replace("_", "-")
        parts.append(k if v is True else f'{k}="{esc(v)}"')
    open_ = f"<{' '.join(parts)}>"
    if name in _VOID:
        return Html(open_)
    return Html(f"{open_}{join(*children)}</{name}>")


def url(path: str, **params: Any) -> str:
    """A local URL; None parameters are left out. Escaped when used in ``tag``."""
    q = {k: v for k, v in params.items() if v is not None}
    return f"{path}?{urlencode(q)}" if q else path


def pre(text: object, *, class_: str | None = None) -> Html:
    return tag("pre", str(text), class_=class_)


def as_json(v: Any) -> str:
    return json.dumps(v, indent=1, ensure_ascii=False, sort_keys=True, default=str)


def details(summary: Child, *body: Child, open_: bool = False, class_: str | None = None) -> Html:
    return tag("details", tag("summary", summary), *body, open=open_, class_=class_)


def table(
    head: Iterable[Child], rows: Iterable[Iterable[Child]], *, class_: str | None = None
) -> Html:
    return tag(
        "table",
        tag("thead", tag("tr", [tag("th", h) for h in head])),
        tag("tbody", [tag("tr", [tag("td", c) for c in r]) for r in rows]),
        class_=class_,
    )


def kv(pairs: Mapping[str, Child] | Iterable[tuple[str, Child]]) -> Html:
    items = pairs.items() if isinstance(pairs, Mapping) else pairs
    return tag("dl", [[tag("dt", k), tag("dd", v)] for k, v in items], class_="kv")
