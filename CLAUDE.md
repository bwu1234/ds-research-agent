# DS Research Agent project guidance

## Current state

This repository contains plans only. Do not describe planned modules, tools,
commands, evaluations, or integrations as implemented. Start with
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

Proposed stack: Python with typed public APIs, validated boundary schemas,
SQLite for run records, stdio MCP, and Ollama for the model. Pin versions when
D0 lands. There are no executable setup or test commands yet; add verified
commands here when the application is scaffolded.

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

The repository is public. Code, docs, and synthetic fixtures are publishable;
benchmark-derived material is not. Benchmark data, benchmark answers,
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
scores separately from verified success and record limitations honestly.

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
