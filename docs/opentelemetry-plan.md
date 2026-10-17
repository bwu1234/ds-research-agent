# OpenTelemetry tracing plan

Status: proposed, 2026-10-17. No OpenTelemetry instrumentation or exporter is
implemented. This is supporting observability work alongside D5; it does not
change evaluation criteria or close an evidence gap about agent quality.

## Outcome and boundaries

New task runs can produce OpenTelemetry traces and send them to a local OTLP
receiver. An operator can follow model requests, tools, retrieval, sandbox
execution, and verification, and correlate spans with the SQLite ledger.
SQLite remains authoritative for evaluation, provenance, and replay.

Version 1 covers all harness conditions: `no_tools`, `inlined_files`,
`given_files`, `codeagent`, `discovery_agent`, and `end_to_end`. Coverage means
the operations a condition actually executes; baseline runs need no tool spans.
Tracing is disabled by default and operates in the trusted host runner. The
sandbox gains no networking, credentials, or telemetry dependencies.

Historical export, backend-specific dashboards, full content capture, metrics,
and distributed tracing inside Ollama or rag-toolkit are follow-ups. A client
span measures the observed request; it does not reveal provider internals.

## Current integration points and gaps

| Area | Existing code | Required change |
|---|---|---|
| Task lifecycle | `eval/kramabench/baselines.py`, `agent_runs.py`, `codeagent_runs.py`, `discovery_agent.py`, `end_to_end.py` | One common run tracing scope, including cleanup and scoring |
| Model requests | `RecordingClient` in `eval/kramabench/agent_runs.py`; baseline recording in `baselines.py` | Wrap every actual request, including repairs and failures, exactly once |
| Retrieval | `RecordingSearcher` in `eval/kramabench/discovery_agent.py`; initial search in the end-to-end runner | Wrap the actual search and correlate its ledger sequence |
| Tools and programs | `ds_research_agent/agent/loop.py`, `discovery.py`, `end_to_end.py`; comparator executor in `codeagent_runs.py` | Explicit tool scopes and child execution scopes around operations |
| Provenance | `LedgerRecorder`, session audits, submission and verification callbacks | Bounded audit attributes and explicit execution identities |
| Storage | `ds_research_agent/ledger/store.py`, schema version 4 | Add durable span correlation and timing without changing replay inputs |
| Configuration | `ds_research_agent/config/settings.py`, `config/example.yaml` | Validated optional telemetry settings |

Runs have start/end timestamps, while steps and programs mostly have durations.
The viewer reconstructs tool placement using conversation and loop semantics.
Exact timing and parentage must be captured during execution for new traces.
Do not reuse the viewer's cumulative model-request timing as a real timeline.

## Span contract

Use one trace per task attempt. A batch is an attribute shared by its runs;
avoid a batch span that stays open for hours. A resumed or replayed execution
gets a new trace, with an explicit reference to the original run where known.

Existing ledger run IDs are `batch/task/repeat`, not attempt identities. Resume
deletes unfinished run rows and reuses their IDs. Allocate a fresh opaque
attempt ID and trace ID for every execution, plus an opaque batch ID mapped
locally to the ledger batch. Keep an additive attempt table containing the
local run/batch mapping, execution mode, predecessor attempt when known, and
lifecycle state. Spans belong to attempts, not directly to replaceable run
rows. Preserve interrupted attempts and their spans when `discard_unfinished`
deletes evaluation rows; neither cascade-delete them nor let their foreign
keys block resume. A lookup by run ID alone must never merge attempts. An
attempt without a recorded completion remains incomplete, including after
process death; do not synthesize a successful root span on resume.

Set execution mode to live or replay on every trace. Replay boundaries measure
the replay itself, not original inference or sandbox latency. Mark replayed
model usage and execution results as recorded evidence; do not report them as
new provider usage or actual sandbox execution. Retain an original-attempt
reference when available, with a local original-run reference when the source
predates telemetry. Replayed operations use explicitly named replay spans
rather than live model-client or sandbox-execution spans.

Illustrative tree (operations occur only where applicable):

```text
task run
  initial catalogue search
  model request
  execute tool: run_python
    select files / restart kernel
    sandbox cell
  model request (including separate requests for repair)
  execute tool: submit_answer
    submission check
  final verification
    fresh sandbox execution
  scoring
```

Tool execution follows a model response, so model calls and tools are siblings
under the run. Correlate tools with the originating request, loop step, call
index, and tool-call ID when provided. Nested execution and verification spans
follow the actual scopes. Cleanup executes before the root span closes.
Add session setup/close spans where they occur; do not invent an execution span
for a rejected tool call. Record validation rejection as a bounded event.

Use standard resource attributes for service identity and a pinned revision of
the GenAI conventions for applicable model and tool attributes. Keep the
mapping in one module and record its version. Use `dsra.*` attributes for
project concepts. The implementation gate selects exact names and span kinds
against the pinned conventions, including any conventions still in development.

Required metadata:

- Run: opaque attempt and batch IDs, execution mode, predecessor/original
  attempt reference when known, condition, repeat, repository revision,
  config hash, stop reason, and a separate answer/evaluation outcome.
- Model: provider/model, request sequence, token usage when supplied, finish
  reason, repair attempt, and observed request timing.
- Tool/execution: tool name, originating request and call identity, program
  sequence/session identity when present, execution status, timeout, observed
  read count, and audit completeness.
- Retrieval: operation, scope, requested/result counts, catalogue generation,
  and ledger sequence when available.
- Verification: reproduction, audit completeness, and verification outcome as
  separate values. A wrong answer is an evaluation result, not automatically
  an OpenTelemetry instrumentation error.

Record UTC epoch timestamps in nanoseconds for span boundaries and use a
monotonic clock for elapsed durations and budgets. Async context must be
isolated between runs. Test context propagation across the comparator's
`asyncio.to_thread` and `run_coroutine_threadsafe` bridge as well as ordinary
async calls, including cancellation and cleanup. Cancellation closes active
spans and preserves cleanup; abrupt process death may leave an incomplete
trace and must not be reported as successful completion.

## Content and delivery contract

Version 1 exports an allowlist of metadata. Prompts, responses, reasoning,
code, search queries/snippets, file paths, file contents, answers, and raw
exception messages stay in existing private artifacts. Export bounded error
types/codes; automatic exception recording must not bypass this policy.
Export opaque attempt and batch IDs to join through the local mapping; do not
export existing structured run IDs, task keys, or user-supplied batch names.
Operation sequence numbers join to ledger rows within that attempt. Apply
the allowlist to resource attributes, span names, events, and status
descriptions as well as span attributes. Benchmark-derived telemetry
stays local and ignored by Git under the existing data policy.

Use the OpenTelemetry Python API/SDK and the official OTLP HTTP/protobuf
exporter. Select and pin versions compatible with Python 3.14 before adding
dependencies, preferably in an optional telemetry extra. Use a no-op facade
when disabled, including when the extra is absent. Explicitly enabled tracing
with a missing dependency or invalid configuration fails before a batch starts.

Add a validated `telemetry` settings section using the existing YAML and
`DSRA_` precedence. Configure enablement, a loopback endpoint, service name,
bounded persistence/export queue and batch sizes, SQLite lock timeout,
exporter request timeout, and total shutdown budget.
For v1, reject remote receiver endpoints. Expose the effective non-secret
settings in batch metadata; avoid independent environment auto-configuration
silently overriding project settings. Sample all spans for enabled runs.

Export runs asynchronously with bounded memory. An unavailable receiver,
queue overflow, or flush failure produces a visible diagnostic and delivery
status; it must not replace the task's result or stop remaining tasks. Delivery
is best effort, with no exactly-once claim. Startup configuration errors and
runtime delivery failures have different handling.

Maintain separate counts/status for local persistence, export attempts,
exporter-reported success/failure, queue drops, and spans still pending or of
unknown delivery at shutdown. Exporter success is not proof of backend
storage or query visibility; receiver-observed evidence is a separate check.
Expose a bounded diagnostic and end-of-command summary even when SQLite
recording fails. Select supported accounting hooks against the pinned SDK;
do not assume its batch processor provides application-level loss accounting.

Configure the HTTP exporter's own request timeout and verify retry behavior;
do not rely on the batch processor's export-timeout parameter to bound an
in-flight request. One total shutdown deadline covers persistence draining,
export flushing, worker joins, and SDK/exporter teardown, including automatic
exit hooks. Record undrained work as incomplete when that deadline expires.
Prove this with a stalled receiver and full queues before expanding coverage.

Persist each completed span's identity, parent identity, timestamps, name,
kind, status, events, and allowed attributes in an additive telemetry table,
plus resource/scope/mapping identity sufficient for later export. This table
also covers operations with no existing Step or Program row. Correlate ledger
rows through the local attempt mapping and sequence attributes rather than
changing request hashes. Only enabled new runs populate it.

Use a dedicated persistence worker with its own SQLite connection and a
bounded queue of immutable, allowlisted records. Never reuse `Ledger._db` from
an SDK or comparator worker thread. Enqueue without waiting for disk or network;
bound SQLite lock waits and retries, and report queue overflow or write failure
without raising into agent execution. Serialize attempt creation before its
span records in that queue. Local recording and OTLP delivery are independent:
a failed persistence queue must not imply that export failed, or vice versa.
During shutdown, stop producers after run cleanup and root closure, drain the
writer and exporter within the shared deadline, then close their resources.
Do not close a connection from another thread while its writer is active.
Test lock contention, write failure, overflow, and shutdown ordering.

Batch export stays off the execution path; a local persistence failure is
reported explicitly without turning it into an agent failure. Missing spans
remain a visible completeness limitation.

## Delivery sequence and gates

### 1. Capture spans and preserve their correlation

Add `ds_research_agent/telemetry/` with the no-op facade, SDK lifecycle,
attribute mapping, and local recording adapter. Add validated configuration,
optional dependencies, and an additive ledger migration. Instrument a
synthetic given-files run first. Defer broad condition coverage until the
persistence and OTLP failure gates below pass.
Start scopes around execution, rather than estimating boundaries in callbacks.

Gate: deterministic recorded-response fixtures show correctly nested spans,
exact request counts through repair, durable IDs/timestamps, cancellation and
failure outcomes, and isolated async runs. Test interrupted-attempt retention,
run-ID reuse on resume, distinct replay identities and timing semantics, and
persistence lock/write failures and bounded queue loss. Old ledgers remain
readable and replayable. With tracing on/off, request hashes, tool results,
scores, and replay behavior match. No network or real model is required for
this gate.

### 2. Export through OTLP and demonstrate local ingestion

Wire the official exporter and bounded SDK batch processor. Provide one
documented local receiver/viewer setup with a pinned image and loopback-bound
ports, and one self-contained synthetic demonstration command that preserves
its ledger, delivery summary, and received trace evidence. Choose the receiver
by a compatibility smoke check rather than adding a backend abstraction.

Gate: the receiver accepts a trace with the expected root, children, resource
attributes, token counts, and error statuses. Evidence distinguishes a span
recorded locally from a span received by the receiver. A receiver outage,
queue saturation, and shutdown timeout leave task outcomes intact and expose
delivery problems. Include a receiver that accepts connections but stalls,
full persistence and export queues, and actual process exit within the total
shutdown budget. Verify the pinned SDK's accounting hooks and exporter timeout
and retry behavior; a flush return value alone is insufficient evidence.
Inspect actual wire payloads for content exclusions, including structured
run/task IDs and batch names. These gates must pass before expanding to all
conditions.

### 3. Finish coverage and measure impact

Add a condition-by-operation coverage matrix and synthetic coverage for all
six conditions, including discovery selection, kernel restarts, comparator
execution, submission checks, final reruns, and session cleanup. Use one shared
span owner per operation to prevent duplicate instrumentation.
Explicitly test comparator parentage through both thread-bridge directions,
deadline cancellation, and cleanup, with no context leaking into the next run.

Measure overhead on repeated deterministic fixtures with tracing disabled,
enabled with a healthy receiver, and enabled with an unavailable receiver.
Record wall time, absolute added time per task and per span, peak memory,
queue loss, and bounded shutdown time. Report steady-state execution and
shutdown costs separately, plus total command time. Proposed
acceptance target: under 5% median wall-time overhead on an agreed repeated
synthetic workload, no task-result changes, and no growth beyond configured
buffer limits. Fast replay fixtures can amplify small fixed costs, so interpret
the percentage alongside absolute overhead; do not silently waive the target
or change the workload after seeing results. Freeze the workload and repetition
count before measuring; do not infer real-model overhead from this measurement.

Gate: offline checks, migration/replay compatibility, receiver ingestion,
failure handling, content policy, and the overhead target all pass. Document
coverage gaps and any abrupt-death limitations. Update the roadmap,
architecture, provenance docs, configuration example, and CLAUDE.md with
implemented behavior and evidence. Run a small development smoke only if
needed to resolve an integration uncertainty; keep holdout sealed.

## Follow-ups after v1

- Export persisted new spans on demand, preserving IDs and observed times;
  define duplicate/retry behavior before offering repeated export.
- Historical conversion: root timing is available, but child timing and
  parentage are reconstructed or absent. Label this explicitly; do not
  present an estimated waterfall as observed timing.
- Full content capture requires a separate local artifact policy, size limits,
  and a deliberate content-export setting. Remote export requires a separate
  policy decision under the benchmark restrictions.
- Propagation into rag-toolkit needs a compatible upstream contract and its
  own instrumentation. This repo can initially show only its client-side span.

## References

- [OpenTelemetry Python instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [Python exporters](https://opentelemetry.io/docs/languages/python/exporters/)
- [OTLP specification](https://opentelemetry.io/docs/specs/otlp/)
- [GenAI span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)

These are reference entry points. Pin exact SDK/exporter and convention
versions in increment 1; the links alone are not a versioned contract.
