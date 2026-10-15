"""Validated application settings.

Settings come from one YAML file, with environment overrides prefixed
``DSRA_`` (nested with ``__``, e.g. ``DSRA_MODEL__THINK=low``). Unknown keys
are errors, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import warnings
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
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ThinkLevel = Literal[False, "low", "medium", "xhigh"]
# File-family rules for group cards (see docs/data-and-provenance.md#group-cards).
GroupRule = Literal["directory", "schema", "directory+schema", "pattern"]


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
    # Ollama server versions (``/api/version``) a model run refuses to start
    # on, because they are known to fail. Any other version runs and is
    # recorded on the batch.
    refuse_server_versions: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_server_version(cls, data: Any) -> Any:
        # ``server_version`` was an exact pin, replaced by the refusal list
        # above. Settings files and ledger rows written before that still
        # carry it; it has no effect now, so it is accepted and dropped.
        if isinstance(data, dict) and "server_version" in data:
            return {k: v for k, v in data.items() if k != "server_version"}
        return data


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
    # Text-like formats (txt, dat, sp3, tle, HTML text, XML fields): lines shown.
    text_preview_lines: PositiveInt
    # CSV rows searched for the real header after a one-cell title row; 0
    # always takes the first row as the header.
    csv_title_scan_rows: NonNegativeInt
    # Rules whose families become group cards; empty writes no groups.
    group_rules: list[GroupRule]
    # Smallest family that gets a group card.
    group_min_members: int = Field(ge=2)
    # A group card lists every member up to this many, then the first and
    # last halves of this many.
    group_member_list_limit: PositiveInt


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
    # Budgets for one agent run. A step is one accepted (or exhausted) model
    # turn; its repair requests do not count separately. Reaching a budget
    # ends the run with that stop reason and no answer.
    max_steps: PositiveInt
    max_wall_s: PositiveFloat
    # Generated tokens (thinking included) per model request (num_predict).
    max_output_tokens: PositiveInt
    # Characters of each of stdout, stderr, and traceback in a tool result.
    max_tool_output_chars: PositiveInt
    # Consecutive identical errors before the tool result demands a re-plan.
    replan_after_repeats: PositiveInt
    # Successful cells in a row whose code is at least ``no_progress_similarity``
    # similar (difflib ratio) to the cell before, before the tool result asks
    # for a change of approach. 0 disables.
    no_progress_after_repeats: NonNegativeInt
    no_progress_similarity: float = Field(ge=0, le=1)
    # From this many steps left, each step's last message carries a fixed
    # notice to commit and submit (the last step: submit now). 0 disables.
    budget_warning_steps: NonNegativeInt
    # Submissions whose program fails in its fresh run, or prints no answer
    # line, that are returned to the model for a fix, per run. Each costs a
    # step; on the last step the submission is final. 0: the first is final.
    submit_checks: NonNegativeInt
    # Identical resends after a model-runner crash, per request (not repairs).
    server_retries: NonNegativeInt


class DiscoverySettings(_Strict):
    """The agentic-discovery condition (D2): search, read cards, choose files.

    Repairs, output tokens, and server retries follow ``agent``."""

    max_steps: PositiveInt
    max_wall_s: PositiveFloat
    # Passages per search_catalogue call (rag_search top_k), before collapse.
    search_passages: PositiveInt
    # Characters of each result's passage shown in search results.
    snippet_chars: PositiveInt
    # Characters of a card read_card returns.
    max_card_chars: PositiveInt
    # Members listed when read_card is given a member filter.
    max_listed_members: PositiveInt
    # Show a grouped file's card as its largest group (D2 progress (c)).
    collapse_groups: bool
    # From this many steps left, the step's last message asks to choose now.
    budget_warning_steps: NonNegativeInt


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
    discovery: DiscoverySettings
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
    model = raw.get("model")
    if isinstance(model, dict) and "server_version" in model:
        warnings.warn(
            f"{path}: model.server_version is no longer used and is ignored; "
            "list versions to refuse in model.refuse_server_versions",
            stacklevel=2,
        )
    return Settings(**raw)
