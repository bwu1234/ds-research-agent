"""D0 acceptance: filtered search over the fixture card index through MCP.

Needs the fixture catalogue built and indexed first:

    uv run python -m ds_research_agent.catalogue --config config/local.yaml build
    uv run python -m ds_research_agent.catalogue --config config/local.yaml index
"""

import time

import pytest

from ds_research_agent.catalogue import load_manifest
from ds_research_agent.config import Settings
from ds_research_agent.retrieval import McpRetrieval, RetrievalError, resolve

pytestmark = pytest.mark.live

QUERY = "identity theft reports by metropolitan area and population"


async def test_filtered_search_resolves_through_manifest(settings: Settings) -> None:
    cat = settings.catalogue
    if not cat.manifest_path.exists():
        pytest.skip("fixture catalogue not built")
    manifest = load_manifest(cat.manifest_path)

    async with McpRetrieval(settings.mcp) as r:
        listed = await r.call("rag_list_corpora", {})
        assert [c["name"] for c in listed["corpora"]] == [cat.corpus]

        t0 = time.monotonic()
        legal = await r.search(
            QUERY,
            corpus=cat.corpus,
            top_k=5,
            max_chars=2400,
            filters={"equals": {"domain": "legal"}},
        )
        cold = time.monotonic() - t0
        t0 = time.monotonic()
        env = await r.search(
            QUERY,
            corpus=cat.corpus,
            top_k=5,
            max_chars=2400,
            filters={"equals": {"domain": "environment"}},
        )
        warm = time.monotonic() - t0

        legal_pairs = resolve(legal, manifest)
        assert legal_pairs and all(e.domain == "legal" for _, e in legal_pairs)
        assert legal_pairs[0][1].file_path.startswith("legal/State MSA/Utah 20")
        env_pairs = resolve(env, manifest)
        assert env_pairs and all(e.domain == "environment" for _, e in env_pairs)

        csv_only = await r.search(
            "station water quality readings",
            corpus=cat.corpus,
            top_k=10,
            max_chars=400,
            filters={"any_of": {"format": ["csv"]}, "equals": {"domain": "environment"}},
        )
        assert {e.file_path for _, e in resolve(csv_only, manifest)} <= {
            "environment/water/station_readings.csv",
            "environment/notes/field_log.csv",
        }

        with pytest.raises(RetrievalError) as e:
            await r.search(
                QUERY, corpus=cat.corpus, top_k=1, max_chars=100, filters={"equals": {"nope": "x"}}
            )
        assert e.value.kind == "tool"

    print(
        f"\nstartup {r.startup_s:.2f}s; cold search {cold:.2f}s; warm search {warm:.2f}s; "
        f"top legal hit {legal_pairs[0][1].file_path} ({legal_pairs[0][1].catalogue_id})"
    )
