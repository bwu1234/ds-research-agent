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


class SandboxSettings(_Strict):
    # Docker CLI; containers run in Docker Desktop's Linux VM on macOS.
    docker: str
    # Tag the runner builds from ds_research_agent/sandbox/image and runs.
    image: str
    # Host directory holding one never-reused directory per sandbox run.
    work_root: Path
    # Wall clock per program, enforced inside the container; the host adds
    # ``host_grace_s`` before it kills the container itself.
    wall_timeout_s: PositiveInt
    host_grace_s: PositiveInt
    # Program limits (RLIMIT_CPU, RLIMIT_FSIZE); container limits below.
    cpu_time_s: PositiveInt
    max_file_bytes: PositiveInt
    memory_mb: PositiveInt
    cpus: PositiveFloat
    pids_limit: PositiveInt
    tmp_mb: PositiveInt
    # Characters of program stdout and stderr returned to the caller each.
    max_output_chars: PositiveInt
    # Persistent kernel: default per-cell limit, the whole session's wall
    # clock, and how long an interrupted cell gets before the kernel is killed.
    cell_timeout_s: PositiveInt
    session_timeout_s: PositiveInt
    interrupt_grace_s: PositiveInt


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


class EvalSettings(_Strict):
    # SQLite run ledger (ignored by Git: it quotes model answers).
    ledger_path: Path
    # Development/holdout split, stratified by domain and difficulty. The
    # split is a pure function of the pinned workload, ``split_seed`` and
    # ``dev_fraction``; ``split_sha256`` freezes it, and runs refuse a split
    # whose hash differs. Null only before the split is first frozen.
    split_seed: int
    dev_fraction: float = Field(gt=0, lt=1)
    split_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Development tasks for the first D3 workflow, one per domain.
    sample_per_domain: PositiveInt
    # Whether prompts state the task's ``answer_type``; recorded on every run
    # and applied identically across conditions.
    answer_type_visible: bool
    # Wall-clock cap per task; a capped run is incomplete and scores zero.
    # Must not exceed model.request_timeout_s, or the client times out first.
    task_timeout_s: PositiveFloat
    # Generated tokens (thinking included) per model request, sent to Ollama
    # as ``num_predict``. A reply cut at the cap ends with done_reason
    # "length" and is recorded as budget_exhausted.
    max_output_tokens: PositiveInt
    # Inlined-files baseline: total characters of file text in the prompt,
    # shared across the task's files (unused shares are redistributed).
    inline_max_chars: PositiveInt
    # Percentile bootstrap over parent tasks, resampled within domain.
    bootstrap_resamples: PositiveInt
    bootstrap_seed: int


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
    sandbox: SandboxSettings
    kramabench: KramaBenchSettings
    eval: EvalSettings

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
