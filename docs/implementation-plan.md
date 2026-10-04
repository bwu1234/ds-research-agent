# Implementation roadmap

Status: all milestones pending. This is a sequence of small deliverables with
acceptance gates, not an estimate of calendar time. Adopted 2026-10-03.

## Goal

This is a portfolio and learning project. Success means a measured agent, a
reproducible evaluation, and an honest write-up. Product adoption is not the
goal.

Given a question and a collection of messy data files, the agent:

1. **Discovers** the relevant files by searching a catalogue of dataset cards
   indexed by rag-toolkit.
2. **Analyses** them by writing and running Python in a sandbox, fixing its own
   errors as it goes.
3. **Answers** with provenance: every reported value carries the files it read
   (with content hashes), the program that computed it, and that program's
   output. A verifier reruns the program in a fresh sandbox, and the value must
   reproduce.

The main benchmark is KramaBench, with DiscoveryBench as an optional addition.
Benchmark facts, splits, and metrics are in [evaluation-plan.md](evaluation-plan.md).

## Model and compute

The main model is `qwen3.8:27b-mlx`, served locally by Ollama. On 2026-10-03,
`ollama show` reported a 27.8B-parameter model, nvfp4 quantisation, a 262,144
token context, and support for tools and thinking (levels false, low, medium,
xhigh; default medium). The installed Ollama version was 0.35.1, and the model
requires at least 0.32.12. It runs on the local Apple M2 Max with 64 GB of
memory. The model name, thinking level, and sampling settings live in
configuration, not code.

Because the model is local, the default checks make no paid calls. The limits
that matter are time and memory:

- **Time budget.** Measure wall-clock time per task in D1. Set per-task and
  per-run time caps in configuration, and report a capped run as incomplete.
- **Memory.** The model (about 18 GB), the rag-toolkit embedding model, and the
  sandbox share 64 GB. Run tasks one at a time until concurrency is measured.
- **Model latency.** Time per step, not cost, is the binding constraint; see
  below.

### Measured model latency

Single rough runs on 2026-10-04 (Ollama 0.35.1, thinking off, generate API):

| Measurement | Result |
|---|---|
| Default loaded context (`ollama ps`) | 262,144 tokens; no implicit truncation at this size |
| Prefill, 24k-token prompt, no cached prefix | about 98 tokens/s (87–250 s before the first output token) |
| Prefill, same 24k prefix with a different suffix | 0.6 s; Ollama reused the cached prefix |
| Generation | about 19 tokens/s |

At these rates, 2,000 thinking tokens take about 100 s, and a cache miss on a
long transcript costs minutes. The loop design follows from this:

- Keep the conversation append-only within a run. Keep the system prompt and
  tool schemas byte-stable, with no timestamps or per-turn state, and never
  rewrite or summarise earlier turns mid-run. Measure how much of the prefix
  survives past assistant turns whose thinking the chat template strips.
- Bound tool outputs (stdout, stderr, table previews) in configuration so a
  single step cannot flood the context.
- Issue one model request at a time. rag-toolkit's base configuration embeds
  queries through the same Ollama daemon (`qwen3-embedding:0.6b`); measure
  whether a search between turns evicts the main model's cached prefix, and
  confirm the served retrieval configuration calls no generative model.
- Choose the thinking level in D1 from measured accuracy and time on the fixed
  development sample, not as a late ablation. It determines whether a full
  development run takes hours or days.
- Record evaluated versus cached prompt tokens and thinking tokens per step.
- **Paid models** are not part of the plan. Any comparison with a hosted model
  is opt-in and needs a declared spend cap before it runs. This also applies
  to benchmark judges; the default scoring profile makes no hosted calls.

## Milestones

| Order | Milestone | Depends on | Result you can report |
|---|---|---|---|
| 1 | D0 Scaffold and integration smoke test | — | Compatible client, model round trip, small fixture index |
| 2 | D1 Evaluation harness and baselines | D0 | No-tools baseline score |
| 3 | D3 Sandbox and analysis with given files | D1 | First complete workflow; development accuracy and runtime |
| 4 | D2 Catalogue expansion and discovery | D1 | File recall per domain and pooled |
| 5 | D4 End to end with provenance | D2, D3 | Development answer scores, verified success, reproduction |
| 6 | D5 Focused ablations and final evaluation | D4 | Paired ablations; frozen holdout results |
| 7 | P1 First-release write-up | D5 | Public results page and synthetic provenance demo |
| 8 | D6 Abstention and calibration (optional) | D4 | Coverage-versus-error curve under a separate frozen protocol |
| 9 | D7 Extensions (optional) | D5 | DiscoveryBench; harder discovery |

D2 and D3 are independent; do D3 first to measure whether the local model can
analyse data within practical budgets before expanding the catalogue. Keep
the milestone IDs stable. The first release is D0–D5 and P1; D6 and D7 are
optional. Retrieval-only D2 checks need no generative model calls. D0 includes
live model checks: a tool-call round trip, a tool-call reliability run, and
prefix-cache measurements.

## D0 — Scaffold and integration smoke test

- Scaffold a Python project with validated configuration, an MCP adapter, and
  a model client interface with an Ollama implementation.
- Pin a rag-toolkit revision and a client compatible with it (see
  [mcp-integration.md](mcp-integration.md)). The local checkout was at
  `3f56da8` on 2026-10-03, ahead of the `fa2a92a` revision inspected for the
  MCP contract. Recheck the contract before pinning.
- Fetch KramaBench with a script pinned to a commit, with checksums. Keep the
  data out of Git. The repository contains answer-bearing directories: place
  only `data/` in the agent-visible store, and `workload/` and `solutions/`
  (reference code per task) in evaluator-only storage. Exclude `dr-input/`
  (per-task copies of files) from the catalogue so files are not duplicated in
  the pooled collection. See [data-and-provenance.md](data-and-provenance.md).
- A deterministic profiler writes one Markdown **dataset card** per file in a
  small synthetic fixture collection (see
  [data-and-provenance.md](data-and-provenance.md)). rag-toolkit indexes the
  cards, not the raw files. Indexing stays a separate job, never an agent tool.
- Confirm that `qwen3.8:27b-mlx` completes one tool-calling round trip through
  the model client. Then run a tool-call reliability check: about 50 synthetic
  tool calls with thinking enabled, measuring malformed calls, wrong argument
  types, and calls to unknown tools. Define a bounded repair-and-retry policy
  and record each retry in the ledger.
- Measure prefix-cache reuse through the model client across a multi-turn tool
  conversation, with and without a rag-toolkit search between turns.
- Spike the trusted read-audit mechanism (see
  [architecture.md](architecture.md#sandbox-requirements)) on a native reader
  (for example, a GDAL-backed geopackage read) and a child process. This is the
  riskiest unbuilt part and the one verified success depends on; D3 still
  makes the final sandbox decision.

Acceptance: discovery and a filtered search work through MCP for the fixture
collection, and returned document IDs resolve through its manifest. Record
pinned versions, fixture index build time, cold/warm search latency, the
model's tool-call round trip and reliability rate, prefix-cache measurements,
and the audit spike's coverage and gaps. Full benchmark catalogue coverage
belongs to D2 and does not block the first sandboxed analysis workflow.

Learn: MCP from the client side, data profiling, what text a retriever needs to
see in order to find a table.

## D1 — Evaluation harness and baselines

- Implement the default local deterministic scoring variant in
  [evaluation-plan.md](evaluation-plan.md), pinning the upstream scorer and
  recording every deviation. Published-score compatibility is a separate,
  opt-in protocol, not a property of the default variant. Report answer scores,
  strict answer accuracy, and verified success separately, with uncertainty,
  time, tokens, and failures in the denominator.
- Split the tasks into development and holdout, stratified by domain, and
  freeze the holdout. Use `legal-tiny` and `environment-tiny` as smoke tests;
  assign their parent tasks to development to prevent overlap. Select a fixed
  handful of development tasks for the first D3 workflow.
- Baselines: (a) no tools, which serves as the contamination probe, and (b) a
  single shot with all of the task's labelled files inlined, truncated to fit
  the context. Record deterministic serialization, truncation, and omitted
  inputs; this is a context-limited baseline, not the given-files code agent.
- Keep answer keys, reference code, and gold sub-task material in evaluator-only
  storage. Verify scoring can replay without network access or credentials.
- Decide whether the agent sees each task's `answer_type`. It is a format
  specification, not the answer, but the choice changes the task and must be
  recorded and applied identically across conditions.
- Choose the default thinking level by running each candidate level on the
  fixed development sample, recording accuracy, thinking tokens, and time.
- Build a throughput model from measured prefill, generation, cache reuse, and
  steps per task, and use it to set per-task budgets and repeat counts.

Acceptance: replaying recorded responses gives identical scores. Both
baselines are reported on the development split, along with measured time per
task and an estimate of how long a full run takes. The `answer_type` decision
and the chosen thinking level are recorded with their evidence.

Learn: evaluating agents when n is small, and why measured variance matters.

## D2 — Catalogue expansion and discovery

- Expand the profiler and index to all 1,764 benchmark files. Each has a card;
  unsupported formats receive a minimal card and an explicit parse status.
  Record coverage, build time, and cold/warm search latency.
- Group cards for file families. Astronomy alone has 1,556 files, mostly CSVs
  likely to share a schema, and `data_sources` uses globs. Per-file cards
  would be near-identical and impractical to select one by one. The profiler
  also writes a group card for each directory or schema family, listing its
  members, and `run_python` accepts discovered group IDs (see
  [data-and-provenance.md](data-and-provenance.md#group-cards)). Measure the
  grouping rule on development tasks before fixing it.
- Retrieval-only metrics: file recall@k and complete-set recall against
  `data_sources`, with globs expanded. A retrieved group card counts as
  retrieving its members; report file-level and group-level results. Report
  per domain and for the pooled collection.
- Establish BM25 and hybrid baselines on a fixed card format. Card-content
  ablations (schema, rows, column values) are optional development experiments.
- Agentic discovery: the model issues several searches and inspects cards
  before choosing files. Compare it with single-shot retrieval.

Acceptance: the retrieval-only scores run offline with no LLM. Scores for the
agentic version are reported with their time cost.

Learn: retrieval over structured data, where text retrieval breaks down.

## D3 — Sandbox and analysis with given files

- A sandbox runner with the guards in [architecture.md](architecture.md). A
  loop that writes code, runs it, reads stdout, stderr, and errors, then fixes
  or finishes. The budget covers steps, tokens, and wall-clock time.
- Start with synthetic fixtures and the fixed D1 development sample. Give the
  agent only the task's labelled files, enforced by sandbox mounts, and build
  the structured answer, ledger, observed data-file read records, and fresh
  rerun before expanding to the development split.
- A persistent per-run Python kernel inside the sandbox for exploration, so
  loaded data survives between `run_python` calls (wildfire alone is about
  1 GB). The final program stays self-contained and is verified in a fresh
  process. The interim alternative, stateless calls caching parsed data as
  parquet in scratch, is acceptable only until the kernel works.
- A sandbox package set derived from the benchmark's file-type inventory: for
  example pandas and openpyxl (csv, xlsx), geopandas/pyogrio (gpkg), numpy
  (npz), and readers for cdf, sp3, and tle files. There is no network, so a
  missing package is a hard capability limit. Record the package list with the
  image and check each format in the inventory loads.
- Loop controls: trimmed tracebacks, bounded outputs, and a forced re-plan
  when the same error repeats. `submit_answer` uses a schema per answer type,
  and the agent runs a sanity check before submitting: magnitude, units, row
  counts after filtering, and nulls.
- A run replay command that renders one run from the ledger (turns, programs,
  outputs, errors, timings), and a fixed failure taxonomy applied to every
  failed development run: discovery miss, parse error, wrong filter or join,
  formatting, budget exhausted, tool-call failure, other. With about 50
  development tasks, reading failures is the main source of improvements.

Acceptance: the small workflow completes from question to scored answer and
fresh rerun, with measured runtime and memory. Then report given-files results
on the development split with uncertainty and a failure-taxonomy count. Sandbox tests block network access,
unselected-file reads, writes outside scratch, and runaway processes. Trusted
read auditing covers native readers and child processes; incomplete auditing
cannot pass provenance verification. Synthetic cases demonstrate that wrong
calculations and hard-coded outputs can reproduce, and exercise the separate
correctness and observed-access checks without claiming causal verification.

Learn: designing execution loops, recovering from errors, isolating containers.

## D4 — End to end with provenance

- Combine D2 and D3. The answer object records the files read (path and hash),
  the final program, its output, and the run ledger.
- Enforce catalogue-selected file access as described in
  [architecture.md](architecture.md); code cannot browse the full data lake.
- The verifier reruns the final program in a fresh sandbox, checks its observed
  data-file reads and hashes, and compares the value. Reproduction does not
  establish semantic correctness.

Acceptance: end-to-end development results alongside given-files results on
the same tasks and analysis configuration. Report answer score, strict answer
accuracy, observed-access verification, reproduction rate, and verified success
separately. The gap measures the effect of this discovery condition, not a pure
causal retrieval cost. Keep the holdout sealed until D5.

Learn: the headline result, and what "traceable" means in practice.

## D5 — Focused ablations and final evaluation

For the first release, make two paired development comparisons against a fixed
D4 baseline, recording time, tokens, and uncertainty:

- How much self-correction to allow, and the stopping policy.
- Agentic versus single-shot discovery.

Profiling, model-generated sub-task planning, further thinking-level
comparisons (the default is chosen in D1), and another local model are
optional later comparisons. Supplying gold sub-task prompts
is an assisted condition and must be reported separately.

Acceptance: retain negative results in the development ablation table. Then
freeze the selected configuration, scoring profile, budgets, repeat count,
baseline comparisons, and report protocol before evaluating the holdout.
Report no-tools, given-files, and end-to-end results on the same holdout tasks.
Do not tune against those results; later adaptive changes need a new untouched
confirmation set or must be labelled exploratory.

## D6 — Abstention and calibration (optional)

The agent may abstain and reports a confidence. Candidate signals include
reproduction and agreement among independently computed intermediate results,
never agreement with gold sub-task answers available only to the evaluator.
These signals need empirical calibration; repeatable wrong answers are possible.

Fit and choose thresholds within development, using an internal calibration
split or cross-validation. If included in the first release, freeze D6 before
D5's holdout evaluation; otherwise use a new untouched confirmation set.
Acceptance: coverage versus error and reliability bins with sample counts.
The holdout is smaller than 104 tasks, so estimates will be coarse.

## D7 — Extensions (optional)

- DiscoveryBench. Its scorer uses a model as judge; fix the judge and check its
  agreement against human labels.
- Harder discovery: add unrelated public datasets to the collection as
  distractors.

## P1 — Write-up

A public page with the architecture, the evaluation setup, given-files versus
end-to-end versus no-tools scores, the ablations, time per task, a provenance
demo using synthetic data, failure analysis, and limitations. Distinguish the
local scoring variant from upstream results. Include no benchmark answers and no
benchmark data, because the KramaBench licence is unclear (see
[evaluation-plan.md](evaluation-plan.md)).

### Deployment

All model, sandbox, and benchmark compute stays local. Published parts are
static or recorded:

- The results page is static (aggregates, methodology, charts) and can be
  hosted at no cost, for example on GitHub Pages or a Cloud Storage bucket.
- An optional read-only replay viewer shows recorded synthetic provenance-demo
  runs: turns, programs, observed reads with hashes, and the rerun result. It
  runs no model or sandbox and serves no benchmark data, so it fits a
  scale-to-zero free tier such as Cloud Run.

Running the agent live in the cloud is out of scope. Checked 2026-10-04
against [Google Cloud's free features](https://docs.cloud.google.com/free/docs/free-cloud-features):
GPUs are never in the free tier and cannot be added during the free trial,
and the always-free VM is one `e2-micro`. The MLX model build runs only on
Apple silicon, Cloud Run's own sandbox would require reworking the sandbox
and read-audit design, and KramaBench data stays local. A live demo would need
paid compute or a hosted model, both opt-in with a declared spend cap, and a
hosted model would differ from the evaluated one.

## Decided

- The repository is named `ds-research-agent`.
- The main model is `qwen3.8:27b-mlx` through Ollama.
- KramaBench data and answers are used locally only. No outreach to the
  authors.
- No agent framework (LangGraph, LangChain) for the core loop; it is written
  in this repository on thin libraries. Decided 2026-10-04 from PyPI and wheel
  inspection:
  - `langchain-mcp-adapters` 0.3.2 requires `mcp<2.0`. The latest 1.x SDK
    (`mcp` 1.30.0) supports protocol revisions only up to `2025-11-25` and has
    no `server/discover`, so it cannot reach rag-toolkit's server, which
    accepts only `2026-07-28`. `mcp` 2.3.0 includes `2026-07-28` and
    `server/discover`.
  - The loop needs byte-stable prompts for prefix-cache reuse and Ollama's raw
    per-step token counts in the ledger; calling the client directly gives both
    without auditing a framework's message formatting and usage mapping.
  - LangGraph's main features (checkpointing, branching graphs, human approval,
    multi-agent coordination) are not needed; the run ledger is the checkpoint
    store.

  Candidate libraries, to pin in D0 once compatibility is shown: `ollama` 0.6.3
  (its `think` parameter accepts the model's string levels), `mcp` 2.3.0,
  `pydantic` and `pydantic-settings`, stdlib `sqlite3`, and `jupyter_client`
  for the sandbox kernel. The `mcp` 2.x client connecting to the real server
  remains D0's compatibility gate. `pydantic-ai` is the fallback if a library
  is wanted; check its MCP SDK version and prompt control first.

## Open decisions

- Development/holdout split ratio, given 104 tasks.
- Sandbox technology: a Docker container or a lighter macOS-native option.
  Decide in D3 from measured isolation, read-auditing coverage, and startup time.
- Read-audit mechanism. On macOS, Docker runs containers in a Linux VM, so the
  auditor must run inside it (for example `strace -f` or fanotify). Informed
  by the D0 spike; decided in D3.
- Whether the agent sees `answer_type`. Decide in D1.
- Group-card granularity: by directory, by shared schema, or both. Decide in
  D2 from development retrieval results.
- Persistent kernel implementation (for example a Jupyter kernel in the
  sandbox) and how its cell history maps to Program records. Decide in D3.
- The judge model and spend cap, only if an upstream-compatible judged
  KramaBench comparison or D7 is explicitly chosen.
