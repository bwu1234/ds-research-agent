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

Historical export, backend-specific dashboards, metrics, and distributed
tracing inside Ollama or rag-toolkit are follow-ups. Content capture of the
model's context window is in version 1 (see the content contract). A client
span measures the observed request; it does not reveal provider internals.

## Current integration points and gaps

| Area | Existing code | Required change |
|---|---|---|
| Task lifecycle | `eval/kramabench/baselines.py`, `agent_runs.py`, `codeagent_runs.py`, `discovery_agent.py`, `end_to_end.py` | One common run tracing scope, including cleanup and scoring |
| Model requests | `RecordingClient` in `eval/kramabench/agent_runs.py`; baseline recording in `baselines.py` | One span per logical request (each repair is one), with one child per send, so server retries and failures each appear exactly once |
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

Every operation a run executes is a span, and every decision the loop makes
without executing an operation is a bounded event, so a trace accounts for
each step of the run. Span tree, from the runners in `eval/kramabench/` and the
loops in `ds_research_agent/agent/` (which spans occur in which condition is in
the coverage table below):

```text
task_run {condition}                 root; attempt identity and outcome
  preflight                          per-run checks before the task starts (Ollama version)
  prepare_inputs                     input manifest, file hashes, inlined-file reads
  retrieval {data_source}            end-to-end seed search, before the agent starts
  invoke_agent {condition}           the agent loop; agent conditions only
    model_step                       one loop step: the request and its repairs
      chat {model}                   one logical request; each repair is another
        model_attempt                one send; a server retry adds another
    execute_tool search_catalogue
      retrieval {data_source}
    execute_tool read_card
    execute_tool choose_files
    execute_tool run_python
      select_files                   resolve IDs, hash check, remount
        sandbox_session_close        a remount ends the live session
      sandbox_session_start          the first cell, and the first after a restart
      sandbox_cell
      sandbox_session_close          the kernel died
    execute_tool submit_answer
      submission_check               argument and file checks
      sandbox_rerun                  fresh container: a submit check, or the final rerun
    workspace_close
      sandbox_session_close
  sandbox_rerun                      comparator: final rerun after the agent returns
  workspace_close                    comparator: after the final rerun
  parse_answer                       baselines: answer extraction
  score
```

In `given_files` and `end_to_end`, the final verification reruns no program.
It reuses the `sandbox_rerun` of the accepted `submit_answer`, so that span
carries the final role, and the verification outcome is an event on the
`submit_answer` span. Only the comparator reruns after the agent returns.
The comparator's model calls are `chat` spans directly under `invoke_agent`,
with no repair. Its code actions are not tool calls, so its setup and action
cells are `sandbox_cell` spans under `invoke_agent`, through the thread bridge.
Its `final_answer` check is its `submission_check`. Workspace cleanup runs
inside the root span and before scoring.

Model steps and tools are siblings under `invoke_agent`, because tools run
after the response that requested them. Correlate tools with the originating
model step and call index. Nested execution spans follow the actual scopes.
Do not invent an execution span for a tool call that did not execute.

Bounded events, with enumerated reasons and counts. The text each decision
adds to the context is on the next `chat` span:

| Event | On | Records |
|---|---|---|
| `tool_call_repair` | `chat` | repair attempt, problem kinds, argument fault kinds |
| `no_tool_call` | `model_step` | cut off by length or plain text |
| `tool_call_not_run` | `model_step` | tool name, call index, reason (already submitted or chosen) |
| `budget_notice` | `model_step` | steps left |
| `replan_notice`, `no_progress_notice` | `execute_tool run_python` | repeat count |
| `kernel_restarted` | `sandbox_cell` | cell status |
| `selection_refused` | `select_files` | problem count |
| `submission_rejected` | `submission_check` | problem count |
| `submit_check_returned` | `sandbox_rerun` | problem kind; the program went back for a fix |
| `choice_rejected` | `execute_tool choose_files` | problem count |
| `verification` | `submit_answer` or root | reproduced, audit complete, access verified |

The stop reason and the answer and evaluation outcome are root attributes.
Command setup, such as settings validation, the Ollama version check, sandbox
image build or reuse, catalogue manifest load, and split verification, gets a
short trace per command. That trace closes before the first task, and each
`task_run` links to it with a span link. This keeps the plan's rule of no batch
span left open for hours.

Coverage by condition:

| Span | `no_tools` | `inlined_files` | `given_files` | `codeagent` | `discovery_agent` | `end_to_end` |
|---|---|---|---|---|---|---|
| `task_run`, `preflight`, `prepare_inputs`, `score` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `chat`, `model_attempt` | ✓ (one) | ✓ (one) | ✓ | ✓ | ✓ | ✓ |
| `parse_answer` | ✓ | ✓ | | | | |
| `invoke_agent` | | | ✓ | ✓ | ✓ | ✓ |
| `model_step` | | | ✓ | | ✓ | ✓ |
| seed `retrieval` | | | | | | ✓ |
| `search_catalogue` + `retrieval`, `choose_files` | | | | | ✓ | |
| `read_card` | | | | | ✓ | ✓ |
| `run_python`, `sandbox_session_*`, `sandbox_cell` | | | ✓ | cells only | | ✓ |
| `select_files` | | | | | | ✓ |
| `submit_answer` | | | ✓ | | | ✓ |
| `submission_check`, `sandbox_rerun`, `workspace_close` | | | ✓ | ✓ | | ✓ |

Replay produces the same tree. Replayed model and sandbox operations use
`replay_` span names in place of `chat`, `model_attempt`, `sandbox_cell`, and
`sandbox_rerun`.

Use standard resource attributes for service identity: `service.name`, plus
`service.version` set to the repository revision. Model, agent, tool, and
retrieval spans follow the GenAI conventions in
`open-telemetry/semantic-conventions-genai`. That repository has no releases
yet and all its GenAI conventions are in development, so pin it by commit SHA.
Keep the mapping in one module and record the SHA. The mapping below is as of
2026-10-17. The implementation gate rechecks each name, kind, and requirement
level against the pinned SHA.

- Inference: `chat {gen_ai.request.model}`, CLIENT, operation `chat`.
  `gen_ai.provider.name` is `ollama`. That is not one of the spec's listed
  values, and the spec allows a custom value. A span covers a logical
  request, including its automatic retries. A server retry
  (`server_retries` in `ds_research_agent/agent/tool_calls.py`) resends the
  identical request, so it is a `model_attempt` child, not another `chat`
  span. A tool-call repair appends messages and is a new logical request, so
  it is a sibling `chat` span under the same `model_step`. The ledger
  records every send as a Step, so each `model_attempt` carries its Step
  sequence.
- Usage: `gen_ai.usage.input_tokens` comes from `prompt_eval_tokens`. That
  count was measured on Ollama 0.35.1 as the whole prompt, including the
  cached prefix, which matches the spec's definition. Re-measure it on the
  running version before relying on it. Ollama reports no cache-read split,
  so leave `gen_ai.usage.cache_read.input_tokens` unset.
  `gen_ai.usage.output_tokens` comes from `output_tokens`, and
  `gen_ai.response.finish_reasons` from `done_reason`, including
  `client_stop`.
- Agent: `invoke_agent {gen_ai.agent.name}`, INTERNAL (an agent in the same
  process), with the condition as the agent name. Not used for baselines,
  scoring, or harness-driven operations such as the seed search and the
  comparator's final rerun.
- Tools: `execute_tool {gen_ai.tool.name}`, INTERNAL, `gen_ai.tool.type`
  `function`. The model client's tool calls carry no ID, so omit
  `gen_ai.tool.call.id` and correlate by step and call index.
- Retrieval: `retrieval {gen_ai.data_source.id}`, CLIENT. The data source is
  a catalogue corpus and domain identifier, never a path, so span names stay
  low-cardinality. The query text (`gen_ai.retrieval.query.text`, opt-in in
  the spec) is recorded under the content contract.
- Leave `gen_ai.conversation.id` unset. The spec forbids substituting a
  generated ID, and no provider conversation exists.
- Errors: `error.type` with bounded values.
- Project spans and project concepts use `dsra.*` attributes.

Required metadata:

- Run: opaque attempt and batch IDs, execution mode, predecessor/original
  attempt reference when known, condition, repeat, repository revision,
  config hash, stop reason, and a separate answer/evaluation outcome.
- Model: provider/model, model step, repair attempt, send attempt and ledger
  Step sequence, token usage when supplied, finish reason, and observed
  timing for each logical request and each send.
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

Version 1 captures content. Everything that enters the model's context
window is explicit in the trace and can be viewed there. That covers:
- system prompt and tool definitions;
- user prompt: the task question, listed files, inlined file text, and seed
  search results;
- every assistant message, with its thinking and tool calls;
- every tool result exactly as appended: rendered cell output after
  truncation, cards, search results, selection text, and replan and kernel
  notices;
- repair feedback, nudges, budget notices, and the request options (`think`,
  `num_ctx`, stop sequences).

The authoritative record is the request as sent. The client records it once
per send, so the trace shows the context the model received, not a
reconstruction of it.
Notices are appended to the last message just before the next request.

The context is append-only, so each `chat` span records only the messages
appended since the run's previous request, plus the count of messages in the
prefix it continues. Messages use the GenAI message JSON form. The system
instructions and tool definitions are recorded once, on the run's first
`chat` span (`gen_ai.system_instructions`, `gen_ai.tool.definitions`), and
later spans carry their digest. Concatenating a run's deltas in order gives
every request's full context. Repeating the whole history on each span (as
`gen_ai.input.messages`) would grow quadratically: measured contexts reach
54k tokens, about 4 MB per 20-step run, past common receiver message limits.
The comparator's messages are recorded as flattened and sent to the client,
not as smolagents' memory.

Each `chat` span records the response as `gen_ai.output.messages`, including
thinking, because thinking returns to the context in later requests. Each
`execute_tool` span records `gen_ai.tool.call.arguments` and the text the
tool returned (`gen_ai.tool.call.result`). Each `retrieval` span records
`gen_ai.retrieval.query.text` and the returned results. Raw cell output beyond
the truncation limit stays in the ledger, because it never reaches the model.

Two parts of the window are outside the client's view. Ollama renders the
messages through the model's chat template on the server. Record the
template's text and digest once per command trace, from `/api/show`, so the
rendering is explicit. Record `prompt_eval_tokens` against `num_ctx` on each
`chat` span, and flag a request that reaches the limit, because the server
may then not see the whole context.

Because the receiver holds this content, telemetry is benchmark-derived
material in full. Its receiver storage lives under the git-ignored data
directory, it is never published, and remote receivers stay rejected.
Exporting content anywhere off this machine is a separate policy decision.
Content attributes are not truncated. Size is bounded by the per-request
delta, not by dropping text, and a record over the exporter's message limit is
reported as a delivery failure, not silently clipped. Gold answers,
reference solutions, and sub-task material are never in the context, so the
trace cannot contain them unless the agent-visible boundary has already
failed.

Export the attempt and batch IDs, and also the ledger run ID and task key so
spans join the run viewer directly. Operation sequence numbers join to ledger
rows within that attempt. Errors carry a bounded `error.type`. Exception
messages go in the span status description, under the same local-only rule.

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
kind, status, events, and attributes, content included, in an additive telemetry table,
plus resource/scope/mapping identity sufficient for later export. This table
also covers operations with no existing Step or Program row. Correlate ledger
rows through the local attempt mapping and sequence attributes rather than
changing request hashes. Only enabled new runs populate it.

Use a dedicated persistence worker with its own SQLite connection and a
bounded queue of immutable records. Never reuse `Ledger._db` from
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
logical-request and send counts through repair and server retry, matched
separately, durable IDs/timestamps, cancellation and
failure outcomes, and isolated async runs. Test interrupted-attempt retention,
run-ID reuse on resume, distinct replay identities and timing semantics, and
persistence lock/write failures and bounded queue loss. Old ledgers remain
readable and replayable. With tracing on/off, request hashes, tool results,
scores, and replay behavior match. A completeness check joins each fixture
run's ledger rows to its spans:
- every Step has exactly one `model_attempt`;
- every Program row (setup, cell, submit check, final rerun, session end) has
  exactly one span of the matching kind;
- every search and selection row has its span;
- the answer, submission, verification, and score rows have their spans or
  events.

The check fails on a missing or duplicate span, and on any span whose kind is
not in the coverage table. A context check rebuilds every request from the
run's `chat` spans, using the prefix deltas, system instructions, and tool
definitions. Each rebuilt request must match the ledger Step's
`request_sha256` byte for byte, so no part of the context window is missing
from the trace. No network or real model is required for this gate.

### 2. Export through OTLP and demonstrate local ingestion

Wire the official exporter and bounded SDK batch processor. Provide one
documented local receiver/viewer setup with a pinned image and loopback-bound
ports, and one self-contained synthetic demonstration command that preserves
its ledger, delivery summary, and received trace evidence. Choose the receiver
by a compatibility smoke check that includes displaying GenAI message
content, rather than adding a backend abstraction.

Gate: the receiver accepts a trace with the expected root, children, resource
attributes, token counts, and error statuses. Evidence distinguishes a span
recorded locally from a span received by the receiver. A receiver outage,
queue saturation, and shutdown timeout leave task outcomes intact and expose
delivery problems. Include a receiver that accepts connections but stalls,
full persistence and export queues, and actual process exit within the total
shutdown budget. Verify the pinned SDK's accounting hooks and exporter timeout
and retry behavior; a flush return value alone is insufficient evidence.
Inspect actual wire payloads to confirm the content reaches the receiver
unclipped and its viewer displays the messages. Run the context check
against the received spans as well as the locally persisted ones, and confirm
the receiver's storage sits under the git-ignored data directory. These gates
must pass before expanding to all conditions.

### 3. Finish coverage and measure impact

Bring the coverage table above to all six conditions with synthetic fixtures,
and run the completeness check on each condition, including discovery
selection, kernel restarts, comparator
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
failure handling, context reconstruction for all six conditions, and the
overhead target all pass. Document
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
- Remote export, or any export of captured content off this machine, requires
  a separate policy decision under the benchmark restrictions.
- Propagation into rag-toolkit needs a compatible upstream contract and its
  own instrumentation. This repo can initially show only its client-side span.

## References

- [OpenTelemetry Python instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [Python exporters](https://opentelemetry.io/docs/languages/python/exporters/)
- [OTLP specification](https://opentelemetry.io/docs/specs/otlp/)
- [GenAI span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)

These are reference entry points. Pin exact SDK/exporter and convention
versions in increment 1; the links alone are not a versioned contract.
