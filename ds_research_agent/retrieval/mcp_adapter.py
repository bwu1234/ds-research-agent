"""MCP client adapter for rag-toolkit's search service.

One adapter owns one server connection. It checks the negotiated protocol
revision and the advertised tools on entry, applies a deadline and a size
bound to every call, and validates the shape of search results. Passage text
is returned as-is; callers must treat it as untrusted evidence.
"""

from __future__ import annotations

import json
import sys
import time
from types import TracebackType
from typing import Any, Literal, Self, TextIO

import anyio
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, ValidationError

from ds_research_agent.config import McpSettings

ErrorKind = Literal["startup", "protocol", "timeout", "tool", "response"]


class RetrievalError(Exception):
    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(f"[{kind}] {message}")
        self.kind = kind


class Passage(BaseModel):
    # Additive server fields are kept, not dropped.
    model_config = ConfigDict(extra="allow", frozen=True)

    rank: int
    score: float
    chunk_id: str
    document_id: str
    source: str
    text: str
    header: str | None = None
    truncated: bool = False
    full_length: int | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    query: str
    corpora: list[str]
    returned: int
    candidate_count: int | None = None
    results: list[Passage]
    hint: str | None = None


class CallRecord(BaseModel):
    """What the adapter observed for one call, for the run ledger."""

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any]
    latency_s: float
    result_chars: int


class McpRetrieval:
    def __init__(
        self,
        settings: McpSettings,
        *,
        server: Any = None,
        errlog: TextIO = sys.stderr,
    ) -> None:
        """``server`` overrides the stdio launch (tests pass an in-process server)."""
        self._settings = settings
        if server is None:
            params = StdioServerParameters(
                command=str(settings.command),
                args=settings.args,
                cwd=settings.cwd,
            )
            # The server's stderr is its log; stdout is the protocol channel.
            server = stdio_client(params, errlog=errlog)
        # The client's own read timeout bounds discovery: an anyio deadline
        # around __aenter__ would exit its scope while the client's task group
        # stays open, which anyio rejects.
        self._client = Client(server, mode="auto", read_timeout_seconds=settings.call_timeout_s)
        self.tool_schemas: dict[str, dict[str, Any]] = {}
        self.startup_s: float | None = None
        self.calls: list[CallRecord] = []

    async def __aenter__(self) -> Self:
        start = time.monotonic()
        try:
            await self._client.__aenter__()
        except Exception as e:
            raise RetrievalError("startup", f"could not connect: {e!r}") from e
        try:
            version = self._client.protocol_version
            if version != self._settings.protocol_version:
                raise RetrievalError(
                    "protocol",
                    f"negotiated {version!r}, configured {self._settings.protocol_version!r}",
                )
            listed = await self._client.list_tools()
            self.tool_schemas = {t.name: dict(t.input_schema) for t in listed.tools}
            missing = sorted(set(self._settings.required_tools) - self.tool_schemas.keys())
            if missing:
                raise RetrievalError("protocol", f"server lacks required tools: {missing}")
        except BaseException:
            await self._client.__aexit__(None, None, None)
            raise
        self.startup_s = time.monotonic() - start
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    @property
    def server_info(self) -> dict[str, Any] | None:
        info = self._client.server_info
        return None if info is None else info.model_dump(exclude_none=True)

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool and return its JSON payload."""
        if tool not in self.tool_schemas:
            raise RetrievalError("tool", f"unknown tool {tool!r}")
        start = time.monotonic()
        try:
            with anyio.fail_after(self._settings.call_timeout_s):
                result = await self._client.call_tool(tool, arguments)
        except TimeoutError as e:
            limit = self._settings.call_timeout_s
            raise RetrievalError("timeout", f"{tool} exceeded {limit}s") from e
        latency = time.monotonic() - start

        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            raise RetrievalError("tool", f"{tool}: {text[:2000]}")
        bound = self._settings.max_result_chars
        if len(text) > bound:
            raise RetrievalError(
                "response", f"{tool} returned {len(text)} chars, over the {bound} bound"
            )
        payload: Any = result.structured_content
        if payload is None:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as e:
                raise RetrievalError("response", f"{tool} returned non-JSON text") from e
        if not isinstance(payload, dict):
            kind = type(payload).__name__
            raise RetrievalError("response", f"{tool} returned {kind}, not an object")
        self.calls.append(
            CallRecord(tool=tool, arguments=arguments, latency_s=latency, result_chars=len(text))
        )
        return payload

    async def search(
        self,
        query: str,
        *,
        corpus: str,
        top_k: int,
        max_chars: int,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        args: dict[str, Any] = {
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
            "max_chars": max_chars,
        }
        if filters:
            args["filters"] = filters
        payload = await self.call("rag_search", args)
        try:
            return SearchResponse.model_validate(payload)
        except ValidationError as e:
            raise RetrievalError("response", f"rag_search payload: {e}") from e
