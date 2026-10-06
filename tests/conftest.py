from pathlib import Path

import pytest

from ds_research_agent.config import McpSettings, ModelSettings


@pytest.fixture
def mcp_settings() -> McpSettings:
    return McpSettings(
        command=Path("/nonexistent/python"),
        args=["-m", "rag.mcp"],
        cwd=Path("/nonexistent"),
        protocol_version="2026-07-28",
        required_tools=["rag_search"],
        call_timeout_s=10,
        max_result_chars=20_000,
    )


@pytest.fixture
def model_settings() -> ModelSettings:
    return ModelSettings(name="m", think="low", request_timeout_s=5)
