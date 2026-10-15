# DS Research Agent project guidance

## Current state

D0 is done: the Python scaffold, validated configuration, Ollama model
client, MCP retrieval adapter, dataset-card profiler, fixture index job,
tool-call repair policy, KramaBench fetch script, and read-audit spike
(`scripts/spike_read_audit/`, not the D3 sandbox) exist. D1's harness exists:
local deterministic scorer, frozen split, SQLite run ledger, no-tools and
inlined-files baselines, replay, and reports (`eval/kramabench/`). D1 is done
(results in `docs/implementation-plan.md`). D3 is done (closed 2026-10-14 in
D3 progress (o)): the sandbox
runner (`ds_research_agent/sandbox/`: restricted container, derived seccomp
profile, fail-closed read audit), its hash-pinned package set, and the
persistent kernel session exist, and so does the given-files agent loop
(`ds_research_agent/agent/`), wired into the KramaBench harness as the
`given_files` condition (`eval/kramabench/agent_runs.py`) and run live on the
smoke tasks and the fixed 12-task sample, with failure-taxonomy labels in the
ledger. The D3 minimal comparator, a modified smolagents `CodeAgent` on the
same sandbox and verifier, is the `codeagent` condition
(`eval/kramabench/codeagent_runs.py`). Both agent conditions have run on all 53
development tasks; the registered paired comparison is decided in
`docs/implementation-plan.md` (D3 progress (k)): execution beats inlined files,
and the custom loop versus `codeagent` is inconclusive. Context stays
append-only (no compaction), and the sandbox records each container's peak
memory. D2 is in progress: profiler version 2 gives 1,741 of the 1,742
benchmark files content cards, each catalogue version has its own config,
corpus, and index, and retrieval-only discovery scoring
(`eval/kramabench/discovery.py`, no model) has BM25 and rag-toolkit baselines
(D2 progress (a)). Profiler version 3 detects CSV title rows and writes
group cards for file families (`ds_research_agent/catalogue/grouping.py`;
rule from an oracle measurement, `eval/kramabench/grouping.py`, D2 progress
(b) and (c)); discovery scoring counts group cards as their members and can
collapse grouped files into their group (`--collapse`). Agentic discovery is
the `discovery_agent` condition (`ds_research_agent/agent/discovery.py`,
`eval/kramabench/discovery_agent.py`): search, read cards, and choose files
by discovered IDs with member filters (`ds_research_agent/catalogue/selection.py`),
scored as `discovery-v1`; its registered comparison with single-shot search
is inconclusive (D2 progress (e)).
Group-ID selection in `run_python` (D4) is not built. rag-toolkit's pooled (unfiltered) search is
not deterministic across runs; domain-filtered search is.
rag-toolkit is pinned at `0f6eb38`. The harness refuses an Ollama version listed in
`model.refuse_server_versions` (0.40.0 to 0.40.3 crash on prefix-cache reuse),
records the version on every batch, stops a batch if the version changes
between runs (auto-update stays on), refuses to resume a batch on a different
version, and warns when `compare` pairs batches from different versions. D1 and
D3 results through (k) were recorded on 0.35.1 and (m) on 0.40.4-rc0; Ollama
runs 0.40.4 since 2026-10-14. `model.options.num_ctx` is set explicitly in `config/local.yaml`; the example
leaves it for the operator to size to their hardware.
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

Verified commands:

```bash
uv run pytest                            # offline checks only (default)
uv run ruff check . && uv run ruff format --check . && uv run mypy
cp config/example.yaml config/local.yaml # then set absolute paths
# Fixture catalogue: profile tests/fixtures/lake, then index it with the
# pinned rag-toolkit clone (rag_service.checkout; must be clean at the pin)
uv run python -m ds_research_agent.catalogue --config config/local.yaml build
uv run python -m ds_research_agent.catalogue --config config/local.yaml index
uv run pytest -m live -s                 # local model + live rag-toolkit server
# D2 benchmark catalogues: a copy of local.yaml per card version, with its own
# catalogue paths, corpus, and rag_service paths (config/local.kramabench*.yaml)
uv run python -m ds_research_agent.catalogue --config config/local.kramabench-v3.yaml build
uv run python -m ds_research_agent.catalogue --config config/local.kramabench-v3.yaml index   # ~15 min; writes index.log
uv run python -m eval.kramabench.discovery --config config/local.kramabench-v3.yaml --ranker rag --mode hybrid --scope domain --out data/measurements/discovery/v3-rag-hybrid-domain.json   # or --ranker bm25; --scope pooled
uv run python -m eval.kramabench.grouping --config config/local.kramabench-v3.yaml --out data/measurements/discovery/grouping-v3.json   # group-rule oracle, ~1 min, no index
uv run python -m eval.kramabench.discovery_paired --pair <a.json> <b.json> --out data/measurements/discovery/paired-<date>.json   # per-task a-b, bootstrap within domain
# D2 agentic discovery (local model + rag-toolkit server; catalogue v3m3, groups of 3+)
uv run python -m eval.kramabench.run --config config/local.kramabench-v3m3.yaml run --condition discovery_agent --tasks smoke --think off
uv run python -m eval.kramabench.run --config config/local.kramabench-v3m3.yaml report --batch <id> --out data/measurements/discovery/agent-<id>.json   # rows for discovery_paired
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
uv run python -m eval.kramabench.run --config config/local.yaml run --condition codeagent --tasks smoke --think off   # D3 comparator
uv run python -m eval.kramabench.run --config config/local.yaml failures --batch <id>   # rule labels, list failed runs
uv run python -m eval.kramabench.run --config config/local.yaml show --run <run_id>     # one run with its gold answer; local only
uv run python -m eval.kramabench.run --config config/local.yaml compare --batch <base> --batch <other> --subset untuned   # D3 primary: the 41 untuned dev tasks (--subset sample: the 12)
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

- `README.md`: scope, status, headline result, and entry points.
- `docs/results.md`: current results and limitations, summarised from the
  plan's dated entries.
- `docs/implementation-plan.md`: milestones, model and compute, decisions,
  dated evidence.
- `docs/architecture.md`: components, tools, sandbox requirements, layout.
- `docs/evaluation-plan.md`: benchmarks, conditions, metrics, splits.
- `docs/data-and-provenance.md`: cards, run records, provenance contract.
- `docs/mcp-integration.md`: current service contract and integration recipe.
- `docs/rag-toolkit-changes.md`: upstream work and ownership.

Update the relevant plan when a contract or design changes. When a comparison
is decided in the plan, update `docs/results.md` and the README headline in
the same change. Keep shared
guidance here; `AGENTS.md` should point here rather than duplicate it.
