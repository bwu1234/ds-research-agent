"""Compatibility gate against the real rag-toolkit server over stdio."""

import pytest

from ds_research_agent.config import Settings
from ds_research_agent.retrieval import McpRetrieval, RetrievalError

pytestmark = pytest.mark.live


async def test_discover_list_and_call(settings: Settings) -> None:
    async with McpRetrieval(settings.mcp) as r:
        assert r.server_info is not None and r.server_info["name"] == "rag-toolkit"
        assert {"rag_search", "rag_list_corpora"} <= r.tool_schemas.keys()
        corpora = await r.call("rag_list_corpora", {})
        assert corpora["corpora"]
        with pytest.raises(RetrievalError) as e:
            await r.search("x", corpus="no-such-corpus", top_k=1, max_chars=100)
        assert e.value.kind == "tool"
        print(f"\nstartup {r.startup_s:.2f}s; server {r.server_info}")
