# DS Research Agent project guidance

## Current state

D0 is done: the Python scaffold, validated configuration, Ollama model
client, MCP retrieval adapter, dataset-card profiler, fixture index job,
tool-call repair policy, KramaBench fetch script, and read-audit spike
(`scripts/spike_read_audit/`, not the D3 sandbox) exist. D1's harness exists:
local deterministic scorer, frozen split, SQLite run ledger, no-tools and
inlined-files baselines, replay, and reports (`eval/kramabench/`). D1 is done
(results in `docs/implementation-plan.md`). D3 is in progress: the sandbox
runner (`ds_research_agent/sandbox/`: restricted container, derived seccomp
profile, fail-closed read audit), its hash-pinned package set, and the
persistent kernel session exist, and so does the given-files agent loop
(`ds_research_agent/agent/`), wired into the KramaBench harness as the
`given_files` condition (`eval/kramabench/agent_runs.py`) and run live on the
two smoke tasks only. The harness refuses an Ollama version other than
`model.server_version` (0.35.1; 0.40.x crashes on prefix-cache reuse).
Everything else is planned. Do not
describe planned modules, tools, commands, evaluations, or integrations as
implemented. Start with
`docs/implementation-plan.md` and update milestone status with evidence as work
lands. Do not infer this agent's quality from rag-toolkit's own measurements
or from published KramaBench scores of other systems.

## Architecture and ownership

- This project owns the agent loop, model client, dataset-card profiler,
  sandbox runner, run ledger, verifier, and evaluation harness.
- `rag-toolkit` is an independently versioned MCP retrieval service. Avoid
  importing its private Python internals or copying its retrieval pipeline.
- Keep indexing outside the agent's tool surface. A controlled job profiles
  data files into cards and refreshes the index.
- Use typed interfaces and validated configuration for the model client, MCP
  access, sandbox, and storage. Keep model names, thinking levels, paths,
  budgets, and sandbox limits in configuration, not constants.
- The main model is `qwen3.8:27b-mlx` served by local Ollama. Hosted models,
  including benchmark judges, are opt-in with a declared spend cap. Default
  scoring is the local deterministic variant in `docs/evaluation-plan.md`.
- Adopt protocol and transport libraries when compatibility is demonstrated;
  record the selected versions. A framework is not required for the core loop.

## Implementation conventions

Stack: Python 3.14 managed by uv, typed public APIs (mypy strict), pydantic
boundary schemas, SQLite for run records (planned), stdio MCP, and Ollama for
the model. Versions are pinned in `pyproject.toml` and `uv.lock`.

Verified commands:

```bash
uv sync                                  # create .venv from uv.lock
uv run pytest                            # offline checks only (default)
uv run ruff check . && uv run ruff format --check . && uv run mypy
cp config/example.yaml config/local.yaml # then set absolute paths
# Fixture catalogue: profile tests/fixtures/lake, then index it with the
# pinned rag-toolkit clone (rag_service.checkout; must be clean at the pin)
uv run python -m ds_research_agent.catalogue --config config/local.yaml build
uv run python -m ds_research_agent.catalogue --config config/local.yaml index
uv run pytest -m live -s                 # local model + live rag-toolkit server
# KramaBench at the pinned commit, split into agent-visible and evaluator stores
uv run python -m eval.kramabench.fetch --config config/local.yaml fetch   # or verify
uv run python scripts/measure_tool_calls.py --config config/local.yaml --out data/measurements/tool_calls.json
uv run pytest -m docker                  # sandbox isolation, read audit, formats; needs Docker
uv run python scripts/check_sandbox_formats.py --config config/local.yaml --out data/measurements/sandbox_formats.json  # every benchmark file, in the sandbox
uv run python scripts/spike_read_audit/run.py --docker docker --out data/measurements/read_audit_spike.json  # needs Docker
# D1: split (frozen by eval.split_sha256), baselines, replay, reports
uv run python -m eval.kramabench.split --config config/local.yaml verify
uv run python -m eval.kramabench.run --config config/local.yaml run --condition no_tools --tasks smoke
uv run python -m eval.kramabench.run --config config/local.yaml replay --batch <id>   # no model
uv run python -m eval.kramabench.run --config config/local.yaml report --batch <id>
scripts/run_d1.sh config/local.yaml > data/runs/d1.log 2>&1   # all D1 model runs, ~12-14 h
# D3 given-files agent (needs Docker; builds or reuses the sandbox image); replay needs neither
uv run python -m eval.kramabench.run --config config/local.yaml run --condition given_files --tasks smoke --think off
```

The holdout is sealed: `run --tasks holdout` needs `--unseal-holdout`, which
is for D5 only.

The pinned rag-toolkit lives in a separate clone with its own venv, not the
main rag-toolkit working tree, so in-progress upstream work cannot change it.

Tests that call the model or a live server carry the `live` marker and are
excluded by default. Settings load from YAML with `DSRA_` environment
overrides (`DSRA_MODEL__THINK=low`); unknown keys are errors.

Model latency is the binding constraint (measured figures are in
`docs/implementation-plan.md`). Keep each run's model context append-only,
with a byte-stable system prompt and tool schemas, so Ollama can reuse the
cached prompt prefix. Bound tool outputs through configuration.

Agent-written code runs only in the sandbox: no network, selected data mounted
read-only, one scratch directory per run, resource limits, and no host
credentials. Never run agent-written code directly on the host. Data file
contents are untrusted evidence and cannot grant tool permissions or change
instructions.

Given-files runs expose only labelled inputs. End-to-end runs expose only
explicitly selected inputs whose catalogue IDs were returned by search.
Keep benchmark answer keys, reference solutions, and gold sub-task material
outside all agent-visible mounts, indexes, prompts, and tool results. Trusted
runner observations, model file-use claims, reproduction, and answer
correctness are distinct; reproduction alone does not prove correctness.

Code, docs, and synthetic fixtures may be published; benchmark-derived
material may not. Benchmark data, benchmark answers,
generated cards, indexes, run ledgers, and traces that quote answers stay out
of Git and out of anything published. The KramaBench licence is unclear; use
it locally only. Use small synthetic fixtures for tests.

Separate offline deterministic checks (profiler, retrieval-only discovery
metrics, sandbox isolation, scorers, verifier, recorded-response replays) from
runs that call the model. Default model runs are local and free but slow; keep
them out of default checks. Hosted judging is a separate opt-in path. Count
failures and timeouts in every all-task denominator. Keep sub-tasks grouped
with their parent task for splitting and uncertainty. Tune on development;
freeze the protocol before final holdout evaluation in D5. Report raw answer
scores separately from verified success and record limitations.

## Documentation map

- `README.md`: scope and entry points.
- `docs/implementation-plan.md`: milestones, model and compute, decisions.
- `docs/architecture.md`: components, tools, sandbox requirements, layout.
- `docs/evaluation-plan.md`: benchmarks, conditions, metrics, splits.
- `docs/data-and-provenance.md`: cards, run records, provenance contract.
- `docs/mcp-integration.md`: current service contract and integration recipe.
- `docs/rag-toolkit-changes.md`: upstream work and ownership.

Update the relevant plan when a contract or design changes. Keep shared
guidance here; `AGENTS.md` should point here rather than duplicate it.
