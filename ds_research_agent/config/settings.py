"""Validated application settings.

Settings come from one YAML file, with environment overrides prefixed
``DSRA_`` (nested with ``__``, e.g. ``DSRA_MODEL__THINK=low``). Unknown keys
are errors, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ThinkLevel = Literal[False, "low", "medium", "xhigh"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(_Strict):
    host: str = "http://127.0.0.1:11434"
    name: str
    think: ThinkLevel
    # Sampling and runtime options passed through to Ollama verbatim
    # (temperature, seed, num_ctx, ...). Recorded per run.
    options: dict[str, Any] = Field(default_factory=dict)
    keep_alive: str = "30m"
    request_timeout_s: PositiveFloat


class McpSettings(_Strict):
    command: Path
    args: list[str]
    cwd: Path
    protocol_version: str
    # Tools the adapter refuses to start without.
    required_tools: list[str] = Field(default_factory=lambda: ["rag_search"])
    call_timeout_s: PositiveFloat
    # Bound on the serialized size of one tool result, in characters.
    max_result_chars: PositiveInt


class CatalogueSettings(_Strict):
    # Agent-visible raw files. The first path component is the domain.
    data_root: Path
    # One Markdown card per file; rag-toolkit's documents_dir for the corpus.
    cards_dir: Path
    manifest_path: Path
    corpus: str
    # Column statistics come from the first ``sample_rows`` rows; rows are
    # still counted to the end of the file.
    sample_rows: PositiveInt
    card_sample_rows: PositiveInt
    max_example_values: PositiveInt
    max_cell_chars: PositiveInt
    # JSON files larger than this get a minimal card instead of being parsed.
    max_json_bytes: PositiveInt


class RagServiceSettings(_Strict):
    # A clean checkout of rag-toolkit at ``revision``; the index job refuses
    # to run against anything else.
    checkout: Path
    revision: str
    python: Path
    base_config: Path
    # Flattened config the index job writes and the MCP server is started with.
    generated_config: Path
    index_dir: Path
    index_timeout_s: PositiveFloat


class AgentSettings(_Strict):
    # Repair turns allowed for one step after a rejected tool call (parse
    # error, unknown tool, or schema-invalid arguments). 0 disables repair.
    tool_call_max_repairs: NonNegativeInt


class KramaBenchSettings(_Strict):
    repo_url: str
    # Full 40-hex commit; the fetch refuses anything else.
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    # Git working copy at ``commit``. Evaluator-only: it holds the answers.
    checkout: Path
    # Agent-visible raw-file store (upstream ``data/`` only). Must not be
    # inside ``evaluator_root`` or ``checkout``.
    visible_root: Path
    # Evaluator-only store (``workload/``, ``solutions/``, scorer code).
    evaluator_root: Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DSRA_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    model: ModelSettings
    mcp: McpSettings
    catalogue: CatalogueSettings
    rag_service: RagServiceSettings
    agent: AgentSettings
    kramabench: KramaBenchSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment wins over the file; no .env or secrets-dir sources.
        return (env_settings, init_settings)


def load_settings(path: Path) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return Settings(**raw)
