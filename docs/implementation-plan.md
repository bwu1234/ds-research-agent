# Implementation roadmap

Status: D0 done (2026-10-06); D1 done (2026-10-08); D3 in progress (sandbox, packages, and kernel done 2026-10-09); all other milestones pending. This is a sequence of small deliverables with
acceptance gates, not an estimate of calendar time. Adopted 2026-10-03.

## Goal

Success means a measured agent, a reproducible evaluation, and a write-up
that reports results and limitations. Product adoption is not the goal.

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
| Prefill, same 24k prefix with a different suffix | 0.6 s; Ollama reused the cached prefix (not reproduced on 2026-10-05 for a change inside the last message; see below) |
| Generation | about 19 tokens/s |

At these rates, 2,000 thinking tokens take about 100 s, and a cache miss on a
long transcript costs minutes.

### Prefix-cache reuse (measured 2026-10-05)

Script: `scripts/measure_prefix_cache.py` (chat API through
`OllamaModelClient`, think `medium`, temperature 0, seed 0, a 4.3k-token
system prompt with three tool schemas, canned tool outputs). Ground truth
comes from the per-request `prefix_cache.go` lines in
`~/.ollama/logs/server.log` (`total`, `matched`, `cached`, `left`); results
JSON stays under `data/measurements/`. Four scenarios, 23 model turns, one run
each.

How the cache works. `qwen3.8:27b-mlx` is a hybrid model (`linear_attn`
recurrent layers plus attention layers). Ollama 0.35.1's MLX runner
(`mlxrunner/prefix_cache.go`, `pipeline.go` at tag `v0.35.1`) keeps a prefix
trie shared across conversations. Recurrent state can resume only where a
snapshot exists: the live end of the last sequence (prompt plus generated
tokens), prompt end minus 4 tokens, every 8,192 tokens of prefill, and branch
points where an earlier request diverged. Paged-out snapshots are kept up to
8 GiB. A divergence anywhere else resumes from the nearest earlier snapshot,
often 0.

| Finding | Evidence |
|---|---|
| `prompt_eval_count` is the whole prompt, not the evaluated part | Identical repeat: 4,298 reported, 0.12–0.14 s (cold: 37–41 s) |
| Changing the end of the last user message re-prefills everything | `matched=4290 cached=388`, 36.9–40.2 s |
| An append-only tool loop resumes after the previous turn's generated tokens | Baseline turns 1–6: `cached` = previous total + previous output tokens; only the new tool result is evaluated (600–3,109 tokens) |
| Thinking passed back verbatim is reused; a second user message does not strip it | Follow-up turn: 24 of 12,087 tokens evaluated, 0.4 s |
| Dropping thinking from history costs one turn's output, not the prefix | Resumes from the prompt-end snapshot (`cached=7436` of 8,079) and re-evaluates about 140 tokens |
| A `rag_search` between turns does not evict the cache | Every turn resumed in full; the embedder runs as a separate `llama-server` runner. `rag_search` loaded only `qwen3-embedding:0.6b` |
| An unrelated request to the same model between turns does not evict the cache | 33-token request, then `cached=7661` and `cached=8477` as in the other scenarios |
| Loading another model under memory pressure does | Another local project's `qwen3.5:9b-mlx` request (4.3 GiB free) stopped the 27b runner twice; the next turn was `cached=0`, 79 s instead of about 6 s |
| Prefill cost tracks new tokens | About 100–115 tokens/s cold and incremental; a 13.9k-token turn cost 16.7 s warm versus 130 s cold |

The loop design follows from this:

- Keep the conversation append-only within a run. Keep the system prompt and
  tool schemas byte-stable, with no timestamps or per-turn state, and never
  rewrite or summarise earlier turns mid-run: any edit before the end of the
  last prompt resumes from the nearest snapshot, usually the start.
- Pass the model's thinking and tool calls back exactly as returned.
- Put task-specific text after a system prompt and tool schemas that are
  identical across tasks; the branch-point snapshot then lets later tasks
  resume after the shared prefix (seen at 388 tokens, where these runs'
  prompts diverged).
- Bound tool outputs (stdout, stderr, table previews) in configuration so a
  single step cannot flood the context; each 1k tokens of tool output costs
  about 10 s of prefill on the next turn.
- Issue one model request at a time, and keep other Ollama models unloaded
  during runs: runner eviction is the one measured way to lose the cache.
  Verifier or side requests to the same model are safe.
- Choose the thinking level in D1 from measured accuracy and time on the fixed
  development sample, not as a late ablation. It determines whether a full
  development run takes hours or days.
- Record prompt tokens, prompt-eval duration, and thinking tokens per step.
  Ollama does not report evaluated versus cached prompt tokens; how the ledger
  records that split is open (see D0 progress).
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

### D0 progress

Done on 2026-10-05, with evidence:

- **Scaffold.** `pyproject.toml` (Python >=3.14, uv), package
  `ds_research_agent` with `config/` (pydantic-settings, YAML plus `DSRA_`
  environment overrides, unknown keys rejected), `models/` (provider-neutral
  `ModelClient` protocol, `OllamaModelClient`), and `retrieval/`
  (`McpRetrieval` adapter). Ruff and `mypy --strict` pass.
- **Pinned libraries:** `mcp` 2.3.0, `ollama` 0.6.3, `pydantic` 2.13.5,
  `pydantic-settings` 2.15.0, `jsonschema` 4.26.0 (added for tool-call
  validation); the lock file records the rest.
- **MCP compatibility gate passed** (live test `tests/live/test_mcp_live.py`).
  The `mcp` 2.3.0 `Client` in `mode="auto"` negotiated `2026-07-28` through
  `server/discover` against rag-toolkit over stdio, listed tools, called
  `rag_list_corpora`, and received a typed tool error for an unknown corpus.
  Startup to tool list took 0.84 s. The first run used the main rag-toolkit
  checkout, which had uncommitted edits; it has since been rerun against the
  pinned clone below (0.95 s).
- **rag-toolkit pinned at `b7434cf`** as a separate clone with its own venv
  (`~/rag-toolkit-dsra`, `pip install -e '.[mcp]'`), so in-progress work in
  the main checkout cannot change the served code. The index job refuses to run
  unless the clone is at the pinned commit with no tracked modifications.
  rag-toolkit has no lock file, so this venv resolved current releases (its
  server runs `mcp` 2.3.0, chromadb 1.5.9, sentence-transformers 6.1.0). Each
  index generation records a hash of the venv's package list and writes the
  list next to the index.
- **Model tool-call round trip passed** (live test
  `tests/live/test_model_live.py`; `qwen3.8:27b-mlx`, think `medium`,
  temperature 0, seed 0). Turn 1 called `add` with correct integer arguments:
  16.6 s wall, 305 prompt-eval and 65 output tokens. Turn 2 answered from the
  tool result: 51.8 s wall, 391 prompt-eval and 40 output tokens. Single run;
  another model (`qwen3.5:9b-mlx`, 11 GB) was loaded concurrently. Turn 2's
  prompt-eval count covering the whole prompt does not show a cache miss:
  Ollama reports the whole prompt either way (see the prefix-cache
  measurement below).
- **Prefix-cache reuse measured** (`scripts/measure_prefix_cache.py`; results
  in [Prefix-cache reuse](#prefix-cache-reuse-measured-2026-10-05)). An
  append-only tool loop with thinking passed back resumes exactly after the
  previous turn's output; searches and unrelated requests between turns do
  not evict it; loading another model under memory pressure does. Open: the
  ledger's evaluated-versus-cached fields have no direct source, because
  Ollama reports only the total. The options are an estimate from
  prompt-eval duration and the measured rate, or joining the server log's
  `prefix_cache.go` lines, which are exact but host-local and
  version-specific.

- **Profiler and fixture collection.** `ds_research_agent/catalogue/`
  profiles `tests/fixtures/lake` (9 synthetic files in 2 domains: CSV with
  `,` and `;` delimiters, cp1252 encoding, nulls, a shared-schema pair, JSON
  arrays and objects, a prompt-injection string, an unsupported `.sp3`, and a
  truncated JSON) into 9 cards: 7 `ok`, 1 `unsupported`, 1 `error`. Builds
  are byte-identical when repeated and are swapped in only after every file
  is profiled. Card schema: [data-and-provenance.md](data-and-provenance.md#d0-card-schema).
- **Fixture index.** `python -m ds_research_agent.catalogue index` writes a
  flattened rag-toolkit config that serves only the card corpus, runs
  rag-toolkit's `index --reset` and `index-report`, checks the document count
  against the manifest and that no chunk is missing or stale, and writes
  `generation.json`. With the embedding model already loaded, indexing 9 cards
  into 10 chunks took 1.35 s.
- **D0 search acceptance passed** (live test
  `tests/live/test_fixture_search_live.py`). `rag_list_corpora` lists only
  `fixture-cards`. `equals` domain filters returned only that domain's cards,
  an `any_of` format filter combined with a domain filter returned only CSV
  cards, an unknown filter field returned a typed tool error, and every
  returned document ID resolved through the manifest to a catalogue entry. The
  top legal hit for an identity-theft query was a Utah MSA card. Latency over
  three runs: MCP startup 0.80–0.96 s; first search after server start
  8.6–15.5 s, which loads the retrievers and cross-encoder (73.5 s on the
  first run in the new venv); next search 0.27–0.33 s. These are 9-card
  fixture numbers and do not predict D2 retrieval quality.

- **Tool-call repair policy** (`ds_research_agent/agent/tool_calls.py`,
  offline tests in `tests/test_tool_calls.py`). A step is accepted when it
  makes no call or every call names an offered tool and its arguments pass
  the tool's JSON Schema (`jsonschema` 4.26.0, Draft 2020-12). Otherwise
  the policy appends feedback and asks again, up to
  `agent.tool_call_max_repairs` (configured 2). Repairs only append: the
  rejected assistant message stays, followed by one tool result per call
  (rejected calls get the error, acceptable siblings are reported as not
  run). Problem kinds follow how Ollama 0.35.1's `qwen3.5` parser
  (`model/parsers/qwen35.go`, `qwen3coder.go`) surfaces failures: a tool
  call whose XML does not parse fails the whole request (`parse_error`; no
  message exists, so only a user feedback message is appended); an unclosed
  `<tool_call>` comes back as content (`unparsed_markup`); `unknown_tool`;
  and `invalid_arguments`, classified as wrong type, missing, unexpected,
  or other. Ollama coerces each value toward its declared schema type
  before we see it, so a wrong type means coercion failed. Each rejected
  response is a `RepairRecord` (attempt, problems, calls, time); writing
  them to the run ledger waits for the ledger (D1/D3).
- **Tool-call reliability run** (`scripts/measure_tool_calls.py`; results in
  `data/measurements/tool_calls_2026-10-06.json`). 50 synthetic single-step
  cases against the planned tool surface (`search_catalogue`, `read_card`,
  `run_python`, `submit_answer`; integer, number, enum, pattern, array, and
  multi-line code arguments) plus 5 requests no tool can serve. Think
  `medium`, temperature 0, seed 0, one run, no other model loaded. Results:

  | Measure | Count |
  |---|---|
  | Malformed calls | 1 of 50 (`parse_error`) |
  | Wrong argument types, missing or unexpected arguments | 0 |
  | Calls to unknown tools | 0 (including "call the plot_chart tool") |
  | Replies with more than one call (rule asks for one) | 2 |
  | Repairs attempted / succeeded | 1 / 1 |
  | Right tool, all argument checks pass | 39 of 50 |

  The malformed call was Python code containing the literal string
  `</parameter>`, which closes Qwen's XML parameter early; after the repair
  feedback the model built the string from parts. Agent code can contain
  that string, so the repair path is needed, not just defensive. Of the 11
  incorrect cases, 9 were `run_python` requests where the model called
  `read_card` first (twice as two parallel calls) instead of running code:
  a schema-valid deviation from the instruction, plausibly good agent
  behaviour, and not a tool-call failure. One search paraphrased `>`/`<`
  into words. Timing: median 5.1 s per call (2.4–17.7 s), median 120 output
  tokens and 210 thinking characters, with the 900-token system prompt and
  tool schemas reused from the prefix cache (median prompt eval 0.67 s).
  These single-step prompts are short and explicit; they bound tool-call
  syntax reliability, not multi-step agent behaviour.
- **KramaBench fetch** (`eval/kramabench/fetch.py`, offline tests in
  `tests/test_kramabench_fetch.py`). Pinned to commit `b2e0d77` (tree
  `f89158b`, upstream `main` on 2026-10-06) in config. A shallow fetch by
  SHA into an evaluator-only checkout (refused unless at the pin with no
  local or untracked changes), then staged copies: upstream `data/` to the
  agent-visible store; everything else except `dr-input/` and `.git` to
  the evaluator store. The stores, checkout, and visible root may not
  overlap. Each store gets a locally generated `SHA256SUMS` (benchmark
  derived, so never committed) and `fetch.json` records commit, tree, and
  counts; `verify` rehashes both stores. The repository has no Git LFS.
  Fetch and split took 77 s. Measured at the pin: 1,742 data files,
  666.9 MB (archeology 5, astronomy 1,538, biomedical 8, environment 37,
  legal 132, wildfire 22); 205 evaluator files; 750 `dr-input/` files
  excluded. This differs from upstream's README table (1,764 files, 1.7 GB,
  wildfire 1 GB); the Hugging Face copy's wildfire directory matches the
  repository, not the README. The domain directory is spelled
  `archeology`.
- **`data_sources` resolution** (reported by `fetch` and `verify`, counts
  only). Entries are relative to `data/<domain>/input` and may be globs.
  Of 314 entries over 106 workload tasks (including the two `-tiny`
  files), 268 match exactly, 35 match only under a subdirectory (legal
  entries omit `csn-data-book-2024-csv/CSVs/`), and 11 do not resolve with
  case-sensitive matching: 5 differ in case (`Identity Theft Data` versus
  `data`; macOS's case-insensitive volume hides this, the Linux sandbox
  will not), 1 typo (`Identitiy`), `omni2.txt` for `omni2.text` (2
  tasks), 1 glob that expects 4 characters after `T` where file names have
  6, and 2 files absent from the repository
  (`WeatherEvents_Jan2016-Dec2022.csv` for wildfire-hard-19, `ZHVI.csv`
  for wildfire-hard-21). D1 must decide how these tasks count; D2 must fix
  a resolution rule for discovery labels.

- **Read-audit spike** (`scripts/spike_read_audit/`; results in
  `data/measurements/read_audit_spike.json`, 2026-10-06). Docker Desktop
  29.8.2 (Linux VM kernel 7.0.14-linuxkit, aarch64), strace 6.13, GDAL
  3.12.4 via pyogrio. Each case runs in a fresh container: `--network none`,
  read-only root, `/data` mounted read-only, one never-reused scratch
  directory, all capabilities dropped except `SYS_PTRACE`, `SETUID`,
  `SETGID`, `CHOWN`, `DAC_READ_SEARCH`, `no-new-privileges`, pids, memory,
  and CPU limits. A root wrapper runs the program as uid 1000 under
  `strace -ff -y` (open-family, exec, and clone-family syscalls) and is the
  only writer of the container's stdout, which carries the trace; the
  program's stdout goes to scratch as its claim. Observed reads are
  successful opens whose returned fd resolves under `/data`. The tracee
  ran with no effective or permitted capabilities (case `tracee_caps`).
  Results were the same with and without `--seccomp-bpf`:

  | Case | Observed |
  |---|---|
  | Python `open`, pandas CSV, GDAL GeoPackage, 50 MB pandas CSV | yes |
  | Child `cat`, child Python, grandchild via nested `sh -c` | yes |
  | Symlink in scratch, `/proc/self/fd` reopen (resolved to the real file) | yes |
  | `mmap`, `shutil.copy` then read the copy (source open seen) | yes |
  | Detached grandchild reading after the program exits | yes |
  | Program kills the tracer | refused (EPERM); read observed |
  | `io_uring_setup` | refused (EPERM) by Docker's default seccomp; not in the trace |
  | `open_by_handle_at` with a real handle | refused (EPERM) by the kernel: tracee lacks the capability |
  | Raw `clone(CLONE_UNTRACED)`, child reads a file | **missed**: the child is not traced; the `clone` call with the flag is in the trace |

  Gaps and caveats:
  - `CLONE_UNTRACED` defeats ptrace following. It is detectable from the
    parent's trace, so the runner must treat any such call as incomplete
    audit. The stronger fix is a seccomp rule that rejects `clone` with
    that flag (and `clone3`, whose flags seccomp cannot inspect, with
    `ENOSYS`), to be decided with the sandbox profile in D3.
  - strace needs `CAP_DAC_READ_SEARCH` to resolve the uid-1000 tracee's
    `/proc/<pid>/fd`; without it every returned fd was undecorated and the
    parser saw no reads. The runner now counts undecorated successful opens
    as audit gaps (0 in this run). Granting the capability also makes
    Docker's default seccomp profile allow `open_by_handle_at`; only the
    kernel's capability check now refuses it.
  - Blocked syscalls refused by seccomp (`io_uring_setup`) do not appear in
    the trace; a custom profile must keep them blocked.
  - Observation is per open, not per byte read, and does not show whether
    the contents influenced the output.
  - Not covered: a persistent kernel traced across many calls (D3),
    syscall numbers other than aarch64, and fanotify or a macOS-native
    sandbox.

  Overhead (median of 3, time inside the container): no-op program 0.02 s
  plain, 0.04 s with `--seccomp-bpf`, 0.09 s ptrace only; pandas on a 5-row
  CSV 0.29 / 0.41 / 0.73 s; pandas on the 50 MB CSV 0.55 / 0.62 / 0.93 s.
  `--seccomp-bpf` is the faster audit mode. Container start-up is excluded
  from these figures; `container_wall_s` in the JSON includes it.

D0 is complete. Group cards are deferred to D2 (`group_count` is 0).

Acceptance: discovery and a filtered search work through MCP for the fixture
collection, and returned document IDs resolve through its manifest. Record
pinned versions, fixture index build time, cold/warm search latency, the
model's tool-call round trip and reliability rate, prefix-cache measurements,
and the audit spike's coverage and gaps. Full benchmark catalogue coverage
belongs to D2 and does not block the first sandboxed analysis workflow.

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

### D1 progress

Implemented on 2026-10-07 (offline checks: `uv run pytest`):

- **Tasks and inputs** (`eval/kramabench/tasks.py`). 104 main tasks and the
  two smoke tasks are loaded from the evaluator store and keyed
  `<workload>/<id>`, because the `-tiny` workloads reuse parent IDs with
  different fields: `legal-tiny/legal-hard-1` has a different answer type,
  sources, and sub-tasks from `legal/legal-hard-1`. Gold answers live in a
  separate field that prompt builders never receive. One resolver, also used
  by `fetch verify`, maps `data_sources` entries in tiers: `exact` (268),
  `nested` (35), `casefold` (5: the case-only mismatches, so a Linux sandbox
  gets the same files as macOS), and `unresolved` (6: a typo, `omni2.txt`
  twice, a glob that matches no file names, and the two absent wildfire
  files). **D0 open item decided:** tasks with unresolved entries stay in
  every denominator. They get the files that do resolve, and each run
  records the unresolved entries. `legal-hard-29` and `legal-hard-30`
  resolve only through `casefold`.
- **Split** (`eval/kramabench/split.py`; open decision closed). The split is
  development 53 and holdout 51, stratified by domain and difficulty (31 hard
  tasks on each side). The quota per domain is round-half-up of half the
  domain; easy and hard share it by largest remainder; seed `20261007`. Smoke
  parents are placed in development first. The split is a pure function of
  the pinned workload files and these settings, so the task lists are never
  committed. `split.json` is written next to `fetch.json`, and
  `eval.split_sha256` (`70fe80e1…`) freezes it: every run recomputes the
  split and refuses a different hash. The fixed development sample for D3
  and the thinking-level sweep is 12 tasks: 2 per domain, alternating easy
  and hard, chosen only from tasks whose sources all resolve. `run` refuses
  the holdout without `--unseal-holdout`.
- **Scorer** (`eval/kramabench/scoring.py`, profile
  `local-deterministic-v1`). The scorer is a port of upstream `metrics.py`
  with every deviation listed. An offline test checks it against the
  pinned upstream file itself, and every gold answer scores 1.0 and strict.
  Details are in [evaluation-plan.md](evaluation-plan.md#scoring-policy).
  Strict `numeric_approximate` tolerance (frozen before any model run):
  `max(1e-6, 1% of |target|)`.
- **Ledger** (`ds_research_agent/ledger/`; SQLite schema 1). Batch, Run,
  Step, Answer, and Score records, described in
  [data-and-provenance.md](data-and-provenance.md#durable-records).
- **Baselines** (`eval/kramabench/baselines.py`). Each makes one request
  with no tools and uses a byte-stable system prompt per condition. The
  answer is the last JSON object with an `answer` key in the reply content,
  parsed with `json` only. *No tools*: the question only. *Inlined files*:
  the resolved labelled files in sorted order. The `eval.inline_max_chars`
  budget (30,000) is shared by water-filling: equal shares, with what short
  files leave redistributed. Text is cut at a line break where possible.
  xlsx sheets are rendered as CSV text (`openpyxl` 3.1.5, read-only),
  because otherwise nearly all of biomedical would be shown no data. gpkg,
  cdf, and npz files (6 references) are listed as omitted. Per-file headers
  are outside the budget (the 124-file astronomy task adds about 17k
  characters). Every run records each file's hash, encoding, and
  shown/truncated/omitted status.
- **Budgets.** Each request is capped at `eval.max_output_tokens` (8,192,
  sent as `num_predict`). A reply cut at the cap is `budget_exhausted`;
  `eval.task_timeout_s` (900) must not exceed `model.request_timeout_s`. All
  failures (timeout, model error, malformed or missing answer, budget
  exhausted, or a planned run with no row) score 0 in every aggregate.
- **Replay.** Each step stores the exact request and its SHA-256 and the
  response as returned. `run replay` re-runs a batch through
  `ReplayClient`, which refuses any request whose hash differs, with no
  model or network, and fails unless every answer and score matches.
- **Reports** (`run report`, `compare`, `choose-think`). These give
  equal-weight task means, strict accuracy, and 95% percentile-bootstrap
  intervals that resample parent tasks within domain (10,000 resamples,
  seed 0). Paired differences use shared resamples. Each report also gives
  per-domain and per-type scores, stop reasons, time, tokens, and a
  throughput model. Reports print aggregates only.

**`answer_type` visibility: hidden** (`eval.answer_type_visible: false`).
Upstream's system interface (`System.serve_query(query, query_id,
subset_files)` at `b2e0d77`) does not pass the answer type. Keeping it hidden
keeps the task the one upstream systems were scored on, and it is applied
identically in every condition. The cost is formatting failures, which the
D3 failure taxonomy counts. This was decided from the protocol, not measured.
A paired visible-versus-hidden run on the sample would be an optional
development diagnostic.

**Thinking-level rule, registered before the sweep ran.** Inline baseline
at `off`, `low`, `medium`, and `xhigh` on the 12-task sample. Choose the
level with the lowest median wall time whose mean answer score is within
one task's worth (1/12) of the best (`report.choose_think`). The sweep is a
single-shot baseline, not the agent loop; D3 may revisit the level with
loop measurements.

First measurements (2026-10-07):

- **No-tools smoke** (2 tasks, think `medium`): both answers parsed, 61 s
  median per task, about 1,250 output tokens, 22.8 generated tokens/s.
  Replaying it from the ledger gave 0 differences.
- **First inline smoke** (60,000-character budget, no output cap): the
  environment smoke prompt was 34,023 tokens. CSV text measured about 1.8
  characters per token, not the 3 assumed. Prefill took about 6 minutes;
  `medium` thinking then ran until the 900 s client timeout. The request
  was not streamed, so nothing partial was recorded. That run led to the
  30,000-character budget, the 8,192-token output cap, and the
  timeout-consistency check.

`scripts/run_d1.sh` runs the sequence (resumable, `--resume` on every batch):
smoke, sweep, the level choice, both development baselines, then a replay
check of every batch. It ran on 2026-10-07 and 2026-10-08, with one
deliberate stop and resume. The interrupted task was discarded and redone.

**Results** (Ollama 0.35.1, `qwen3.8:27b-mlx`, temperature 0, seed 0, one
run per task, repository uncommitted at run time so ledger rows record a dirty
tree, scoring profile `local-deterministic-v1`, `answer_type` hidden, 8,192
output-token cap). Aggregates only; intervals are 95% percentile bootstrap
over parent tasks within domain.

*Thinking-level sweep* (inlined files, 30,000-character budget, the fixed
12-task sample). The pre-registered rule chose **thinking off**.

| Level | Answer score | Answered / hit cap | Median wall | Median output tokens |
|---|---|---|---|---|
| off | 0.333 | 7 / 5 | 414 s | 5,405 |
| low | 0.167 | 5 / 7 | 484 s | 8,192 |
| medium | 0.167 | 6 / 6 | 463 s | 7,265 |
| xhigh | 0.167 | 6 / 6 | 487 s | 8,024 |

The evidence is weak: the margin is two tasks of 12, and about half of all
runs at every level hit the output cap. Thinking levels spent more tokens
and did not improve the answers in this single-shot setting. It is a
single-shot baseline, so D3 must recheck the level inside the agent loop.

*Development split, 53 tasks, thinking off:*

| Baseline | Answer score (95% CI) | Strict accuracy (95% CI) | Answered / cap / malformed | Wall: median, mean | Full dev run |
|---|---|---|---|---|---|
| No tools | 0.090 (0.036–0.155) | 3.8% (0–9.4%) | 49 / 3 / 1 | 73 s, 92 s | 1.35 h |
| Inlined files | 0.414 (0.306–0.520) | 35.8% (26.4–45.3%) | 44 / 9 / 0 | 281 s, 291 s | 4.29 h |

Paired difference (inlined minus no tools): answer score +0.324 (0.201 to
0.443), strict accuracy +0.321 (0.208 to 0.434). Verified success does not
apply (no program). The 9 capped inlined runs and the 4 unparseable
no-tools runs all score 0 in these numbers.

By domain (inlined, answer score): legal 0.73, astronomy 0.50, environment
0.40, wildfire 0.36, **archeology 0.00, biomedical 0.00**. Both zero
domains label xlsx files; their sheets are rendered as CSV, truncated to
the budget. The zeros are not diagnosed: the file text may be inadequate, or
the model may have failed on the questions themselves.
No-tools strict successes (2 of 53) are one `numeric_exact` and one
`string_exact`; this probe cannot say whether they are memorisation or
chance.

*Operations and throughput.* Cold prefill measured 103 to 137 tokens/s and
generation 22 to 23 tokens/s. A full dev run is 1.35 h no-tools and 4.29 h
inlined at this budget; the same figures put the holdout at 1.30 h and
4.13 h. A D3 agent step, at 1,000 tool-output tokens and the inlined
batch's median output, comes to about 155 s, so 10 steps are about 26
minutes per task and the 53-task development split would take about 22 h
at 10 steps per task. D3 must choose per-step output and step caps with
this in mind. The model-throughput model is in the report's `throughput`
block.

*Replay.* All 7 batches (smoke, four sweep levels, both development
baselines; 144 runs) replay from recorded responses with 0 differences in
answers and scores.

Limitations: one run per task and no repeat, so run-to-run variance is
unmeasured. The sweep has 12 tasks. A third of inlined runs hit the output cap,
so the inlined score is a floor for what this model could do with more
tokens. Whether a higher cap would change the picture is not measured
(a diagnostic rerun of the capped sample tasks at 16,384 tokens is
optional and not run). The archeology and biomedical zeros are undiagnosed.

D1 acceptance (replay identical; both baselines on the development split
with time per task and a full-run estimate; `answer_type` and thinking level
recorded with evidence) is met.

## D2 — Catalogue expansion and discovery

- Expand the profiler and index to all 1,742 benchmark files. Each has a card;
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

## D3 — Sandbox and analysis with given files

- A sandbox runner with the guards in [architecture.md](architecture.md). A
  loop that writes code, runs it, reads stdout, stderr, and errors, then fixes
  or finishes. The budget covers steps, tokens, and wall-clock time.
- Start with synthetic fixtures and the fixed D1 development sample. Give the
  agent only the task's labelled files, enforced by sandbox mounts, and build
  the structured answer, ledger, observed data-file read records, and fresh
  rerun before expanding to the development split.
- A persistent per-run Python kernel inside the sandbox for exploration, so
  loaded data survives between `run_python` calls (astronomy alone is about
  500 MB). The final program stays self-contained and is verified in a fresh
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

### D3 progress

Order of work: (a) sandbox runner, (b) package image from the file-type
inventory, (c) persistent kernel, (d) agent loop and `submit_answer`,
(e) fixtures, then the 12-task sample, then the development split.

- **(a) Sandbox runner, done 2026-10-09** (`ds_research_agent/sandbox/`).
  `SandboxRunner.run(program, inputs)` runs one Python program in a fresh
  container and returns its exit code, bounded stdout and stderr, timing,
  and the trusted read audit with each observed container path mapped back
  to its host file. It generalizes the D0 spike's container: `--network
  none`, read-only root, one read-only bind mount per selected input under
  `/data` (unselected siblings do not exist), a writable scratch directory
  (reusable across calls in one run), program and output in separate
  per-call mounts, all capabilities dropped except the four the root
  wrapper needs for `strace -u`, `no-new-privileges`, `--ipc none`, and
  memory (swap disabled), CPU, pids, and `/tmp` limits. The program runs
  as uid 1000 under `strace -ff -y --seccomp-bpf` with `prlimit` CPU-time
  and file-size limits that do not apply to the tracer; the wrapper enforces
  the wall clock and kills every remaining process before it prints the
  trace. Limits and paths are in the `sandbox` settings section.
  - **Seccomp profile** (`sandbox/image/seccomp.json`, generated by
    `scripts/build_seccomp_profile.py` from Docker 29.8.2's default profile,
    pinned by Git blob hash). The only change adds `CLONE_UNTRACED` to the
    mask of `clone` flags that fall through to `EPERM`. Docker's default
    already returns `ENOSYS` for `clone3`, and libc falls back to `clone`.
    Closes the D0 spike's one gap.
  - **Audit fails closed.** The audit is incomplete when the end marker is
    missing, no process was traced, the wall clock expired, a successful open
    has an undecorated fd, an open line does not parse, or a `clone` with
    `CLONE_UNTRACED`, `io_uring_setup`, or `open_by_handle_at` succeeded.
    Successful `execve` of a data file counts as a read.
  - **Image** pinned by base digest (`python:3.14-slim`, Debian 13.7,
    Python 3.14.8), strace 6.13, and an exact `--no-deps` requirements list
    checked by `pip check`: pandas 3.0.6, numpy 2.5.3, pyogrio 0.13.0,
    openpyxl 3.1.5 and their dependencies. `image_info()` records the image
    ID and `pip freeze`. Not yet reproducible from scratch: the Debian
    packages (strace) are not snapshot-pinned and pip installs are not
    hash-pinned.
  - **Tests.** `tests/test_sandbox_audit.py` (offline, default run, 12
    parser cases). `tests/sandbox/test_isolation.py` (marker `docker`, 18
    tests, about 18 s): no route and no TCP or DNS; only the selected
    inputs exist and host paths, the Docker socket, and `~/.ssh` are absent;
    writes succeed only in scratch and `/tmp`; no host environment; uid 1000
    with no effective, permitted, inheritable, or ambient capabilities and
    `NoNewPrivs`; wall-clock timeout (audit incomplete), CPU, memory,
    process-count, and output limits; reads observed for pandas, a nested
    shell child, a detached grandchild, and GDAL (GeoPackage via pyogrio);
    `CLONE_UNTRACED` refused with `EPERM`; `clone3` `ENOSYS` while threads
    and `subprocess` still work; the program cannot kill the tracer; mount
    validation.
  - Checked by hand: with Docker's stock profile the `CLONE_UNTRACED` child
    runs and its read is missed, and the audit reports it incomplete. So the
    profile and the parser each catch it.
  - Seccomp refusals by Docker's filter do not reach strace (its `ERRNO`
    outranks strace's `TRACE`), so blocked attempts are not in the trace.
    `blocked_escapes` records only refusals the tracer sees.
  - **Startup** (Docker Desktop 29.8.2, kernel 7.0.14-linuxkit, 10 runs
    each): an empty program takes a median 0.17 s of wall time (0.15 to
    0.20 s); importing pandas and reading a small CSV takes 0.56 s (0.49 to
    0.72 s). This is negligible next to the ~155 s estimated D3 model step.
  - Not done in (a): input-file hashes (D4's answer object), recording
    sandbox runs in the ledger (with the loop in step d), and an OOM-kill
    flag (memory exhaustion shows as a nonzero exit code only).
- **(b) Package set, done 2026-10-09.** File-type inventory of the
  fetched agent-visible store (1,742 files, as listed in its `SHA256SUMS`;
  the evaluation plan's per-domain table counts 1,764 from the upstream repo
  tree, and the two have not been reconciled): csv 1,626; sp3 37 with 37 XML `.HDR`
  sidecars; txt 18; xlsx 10; tle, lst, gpkg, dat 2 each; text, py, npz,
  json, html, cdf 1 each. By domain: astronomy 1,538 (every non-CSV format
  except xlsx, gpkg, json, html, py, and one txt), legal 132, environment 37,
  wildfire 22, biomedical 8, archeology 5.
  - Top-level packages in `sandbox/image/requirements.in`, compiled by
    `uv pip compile --universal --generate-hashes --exclude-newer
    2026-09-25` (only releases at least two weeks old) into
    `requirements.txt`. The image installs wheels only, with
    `--require-hashes`, and pins `strace=6.13+ds-1`, so the build fails
    rather than drifting. This closes step (a)'s reproducibility gap,
    except that Debian packages still come from the live mirror.
  - Format readers: pandas (csv and whitespace tables), openpyxl (xlsx),
    pyogrio and geopandas with shapely and pyproj (gpkg), numpy (npz),
    cdflib 1.3.14 (cdf), sgp4 2.27 (tle), lxml (`read_html`, XML), pyarrow
    (parquet in scratch). sp3 has no maintained reader (`sp3` 1.1.1 is from
    2022 with no wheels; `georinex` was last released in 2023), so it is
    parsed as fixed-format text. Also included, as a capability choice rather
    than a format need: scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1.
    29 packages in all; the image is 1.22 GB.
  - `tests/sandbox/test_formats.py` (marker `docker`) writes a synthetic file
    per format in one sandbox run and reads each back as a mounted input in
    another (sgp4 propagates a public ISS TLE). The audit observes every
    file.
  - `scripts/check_sandbox_formats.py` (local only; per-file results under
    `data/measurements/`) loaded every agent-visible file in the sandbox, one
    run per domain, about 54 s in all (biomedical's xlsx files took 40 s).
    Every file was readable with the image's packages. Status: cdf, dat,
    gpkg, hdr, html, json, lst, npz, sp3, tle, xlsx all `ok`. CSV: 1,611
    `ok`, 10 need latin-1, and 5 fail pandas' default parser with ragged
    rows (more fields than the header). Those are wrangling problems, not
    missing packages. The 18 txt files, the one `text` file, and the `py`
    file are text that is not a whitespace table. All six domain runs had a
    complete audit, and every loaded file was observed.
- **(c) Persistent kernel, done 2026-10-09** (`sandbox/session.py`,
  `sandbox/image/bridge.py`, `sandbox/image/kernel.py`). `KernelSession`
  keeps one container per agent run, with the same isolation as program
  runs plus `--init` to reap orphans. The channel is the container's
  stdin and stdout (`docker run -i`): with `--network none`, ZMQ over TCP is
  out, and a Unix socket on a bind mount does not cross Docker Desktop's VM
  (measured: the host connect was refused). Chosen over ipykernel with
  `jupyter_client`, which would add pyzmq, tornado, and IPython to the image
  and more processes under strace, for text-only output.
  - **Trust split.** A root bridge is the only writer of the container's
    stdout. It relays one JSON request per line to the kernel (uid 1000,
    under strace for its whole life) and returns each cell's result with the
    trace lines written during the cell, its wall time, and its stdout and
    stderr. Those are captured at the fd level into per-cell files, so child
    process output is included. The kernel's own status line is untrusted. A
    reply for the wrong cell is a `protocol_error`, and the kernel is killed;
    a test forges a reply to check this.
  - **Cells** run in a fresh `__main__` module (so cell-defined functions
    pickle). A trailing expression's repr is printed. Tracebacks start at the
    cell's first frame. `SystemExit` does not end the kernel. Between cells
    fds 0 to 2 are `/dev/null`, so a cell cannot read requests from stdin.
  - **Limits.** Per-cell `cell_timeout_s` sends SIGINT, which gives
    `timeout_interrupted` with state kept. If the cell does not stop within
    `interrupt_grace_s`, every uid-1000 process is killed (`timeout`, kernel
    dead). Crashes, OOM, and the CPU limit give `dead`. `session_timeout_s`
    caps the whole session. The bridge kills tracees, not strace, so the
    session audit stays complete in all of these cases. Root signalling uid
    1000 needs `CAP_KILL`, now granted to the container (the program still
    has none). Without it, program mode's cleanup `kill -9 -1` had been
    failing silently. That was harmless, because strace exits only after its
    tracees and a timeout already marks the audit incomplete.
  - **Audit attribution.** strace writes each trace line as it happens, so
    per-cell slices attribute reads to the cell during which they happened. A
    background thread's or process's read can land in a later slice. The
    session audit, from the merged trace, is the record; the verifier reruns
    the final program in a fresh container with `SandboxRunner.run` and never
    reuses a session.
  - Mapping to Program records: one cell record per `execute` (code, status,
    outputs, timing, observed reads). The ledger tables arrive with the loop
    in step (d).
  - `tests/sandbox/test_kernel.py` (marker `docker`, 14 tests). Measured
    (Docker Desktop, 5 sessions): start 0.16 s median, a trivial cell's round
    trip 1 ms median (3 ms max), close 0.14 s.

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
demo using synthetic data, failure analysis, limitations, and related work.
Related work places the agent against published systems and the vendor designs
under [Vendor designs](#vendor-designs-checked-2026-10-07). It claims only what
their documentation shows, not how the products behave. Distinguish the
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

## Candidate components from the literature survey

A survey on 2026-10-06 (multi-source search, 25 claims adversarially verified,
13 kept) looked for components a data-science agent usually has that this plan
lacks. None of these is implemented or committed. Each is a hypothesis to test
on the development split, with failures and timeouts in the denominator. The
evidence is almost entirely from 2024-25 papers using hosted models (GPT-4/4o,
Qwen-72B) on small tabular Kaggle-style tasks, mostly ablations of 4-12 tasks
without variance. None measures a 27B local model, and no KramaBench-specific
claim survived verification, so none of it says how much each item would help
here.

| Candidate | Already in the plan? | Home | Evidence (strength) |
|---|---|---|---|
| Statistical-validity layer: deterministic assumption checks (outliers, leverage, distribution), a multiple-comparison counter, and a per-run record of analysis decisions (variables, transforms, model) in the ledger. A robustness pass over alternative specifications is a later, costlier option. | No | D3 checks and ledger fields; the robustness pass in D5 or D7 | Fisher-R1/P-Bench (one Aug 2026 preprint): agents run valid code with flawed inferential choices. BLADE: low coverage of defensible analysis choices (medium) |
| Lifecycle-aware verifier: classify crashes, check return value versus printed output, score format errors separately from reasoning errors, and hash inputs before and after the run. | Partly: hashes and the fresh rerun exist; format-versus-reasoning scoring and crash classes do not | D1 scoring, D3 and D4 verifier | DSEval. In one setup the pass rate rose from 34% to 55% when presentation errors were ignored; weaker models gained more (medium) |
| Bounded iteration: a debug-attempt cap, regenerate from scratch after N similar failures, and a termination check. A task DAG is the larger option. | Partly: forced re-plan on repeated error, budgets, and the D5 self-correction ablation; no explicit caps or DAG | D3 loop controls; caps tuned in the D5 ablation | AutoKaggle (cap 5, regenerate after 3, plateau at 10-15 attempts); Data Interpreter task and action graphs; DA-Code failure modes. The graph ablation does not separate planning from debugging (medium) |
| Explore-first enforcement: the agent must inspect real files, schemas, and dtypes before writing analysis code, with "data not found" and "data misread" as separate failure classes. | Mostly: cards, given-files runs, and the failure taxonomy; no enforced inspection step | D3 loop and taxonomy | DSBench and DA-Code qualitative failure analyses (medium-low) |
| Curated library of pre-validated cleaning tools. The schemas must stay byte-stable for prefix caching. | No | After D3 measurements, if wrangling errors dominate the taxonomy | AutoKaggle: valid submissions 0.58 to 0.88 with cleaning tools; feature-engineering tools added nothing; 4 tasks (medium-low) |
| Compaction policy for long runs, for example an append-only prefix with occasional summary checkpoints. | No: context is append-only by design | Decide in D3 from measured context growth | AIDE summarizes a solution tree instead of appending history. This conflicts with prefix reuse, so it is a trade-off, not evidence against append-only (medium) |
| Retrieval of prior solutions coupled to run feedback. | No | Defer. It needs a case bank built from ledger runs, which raises the contamination risks this project already guards against. | DS-Agent ReviseRank, 12 tasks with GPT-4 (low). Snowflake Cortex Analyst retrieves analyst-verified queries (vendor design, no ablation) |
| Value index for discovery: the index job adds the distinct values of low-cardinality string columns to what `rag_search` can match, so a question naming a specific city, agency, or category finds files whose cards did not sample that value. Never an agent tool. | No: cards list at most `max_example_values` values per column | D2 retrieval-only ablation on file recall (no model calls) | Snowflake Cortex Agents pair semantic views with Cortex Search to match literal values in high-cardinality fields (vendor design, no ablation) (low) |
| Stated interpretation: an `assumptions` field in `submit_answer` listing the interpretation choices the agent made, recorded in the ledger; "had to resolve an ambiguity" then becomes a candidate D6 abstention signal. Overlaps the analysis-decision record in the statistical-validity row. | No | D3 answer schema (before tool schemas are frozen for prefix caching); signal tested in D6 | Snowflake Cortex Analyst's classification agent rejects ambiguous questions rather than answer them misleadingly (vendor design, no ablation) (low) |

Topics with no verified evidence: human-in-the-loop checkpoints, report
generation, uncertainty reporting, and provenance. No vendor engineering
write-ups survived verification in the survey. Absence of evidence is not
absence of need; provenance is already covered by D4 and
[data-and-provenance.md](data-and-provenance.md).

### Vendor designs (checked 2026-10-07)

A follow-up web check of vendor documentation, not adversarially verified, and
of product descriptions rather than measured behaviour. Two shapes dominate:
warehouse agents that generate queries through a semantic layer (Snowflake
Cortex Analyst, Microsoft Fabric data agent, Hex notebook agent), and sandboxed
Python over user-supplied files (ChatGPT data analysis; Julius, from
third-party sources only). Google's Colab Enterprise Data Science Agent plans,
then runs code in the notebook runtime over CSV files and BigQuery tables.
Databricks Genie Code was found only in press coverage. Fabric requires
lakehouse files to be loaded into tables before it can query them. None of the
documentation read records per-value input hashes or reruns the program to
check reproduction; the nearest are Hex citing the cells and projects it used
and ChatGPT showing the code it ran. Vendor accuracy figures are self-reported
on their own task sets and are not comparable with KramaBench. Sources:
[Snowflake engineering blog](https://www.snowflake.com/en/blog/engineering/snowflake-cortex-analyst-behind-the-scenes/),
[Fabric data agent](https://learn.microsoft.com/en-us/fabric/data-science/concept-data-agent),
[Hex notebook agent](https://learn.hex.tech/docs/explore-data/notebook-view/notebook-agent),
[ChatGPT data analysis](https://help.openai.com/en/articles/8437071-data-analysis-with-chatgpt),
[Colab Data Science Agent](https://docs.cloud.google.com/bigquery/docs/colab-data-science-agent).

Sources: Data Interpreter (arXiv 2402.18679), DS-Agent (2402.17453), AIDE
(2502.13138), AutoKaggle (2410.20424), DSEval (2402.17168), DSBench
(2409.07703), DA-Code (2410.07331), BLADE (2408.09667), Fisher-R1/P-Bench
(huggingface.co/papers/2608.07437).

## Decided

- Development/holdout split: 53/51, stratified by domain and difficulty,
  frozen by hash (D1, 2026-10-07).
- `answer_type` is hidden from the agent in every condition (D1, 2026-10-07).
- Sandbox technology: a restricted Docker container (D3, 2026-10-09). It
  passes the isolation tests and starts in about 0.17 s, so a lighter
  option would have little to gain against a ~155 s model step. No
  macOS-native alternative was measured.
- Read audit: `strace -ff -y --seccomp-bpf` under a root wrapper in the
  container, with a seccomp profile derived from Docker's default that
  refuses `clone` with `CLONE_UNTRACED`, and an audit that fails closed (D3,
  2026-10-09; see D3 progress).
- Persistent kernel: a minimal REPL behind a trusted bridge over the
  container's stdio, not ipykernel (D3, 2026-10-09; see D3 progress).


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

- Group-card granularity: by directory, by shared schema, or both. Decide in
  D2 from development retrieval results.
- Which candidate components from the literature survey to build, in what
  order, and whether each needs its own ablation. Statistical checks should be
  deterministic and outside the model unless a local 27B run shows it can do
  them with an acceptable false-positive rate. Decide after D3 failure-taxonomy
  counts exist.
- Long-run context compaction versus strictly append-only context. Decide in D3
  from measured context growth and cache-reuse figures.
- The judge model and spend cap, only if an upstream-compatible judged
  KramaBench comparison or D7 is explicitly chosen.
