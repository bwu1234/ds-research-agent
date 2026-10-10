# Evaluation plan

Status: the D1 harness is implemented (local deterministic scorer, frozen
split, run ledger, no-tools and inlined-files baselines, replay); results
are in [D1 progress](implementation-plan.md#d1-progress). The holdout is
sealed. Define acceptance criteria before scoring a holdout.

## Benchmarks (checked 2026-10-03)

### KramaBench is the main benchmark

Source: [mitdbg/KramaBench](https://github.com/mitdbg/KramaBench), last pushed
2026-07-24, [paper](https://arxiv.org/abs/2506.06541) at ICLR 2026.

| Domain | Tasks | Sub-tasks | Files | Raw size | File types (from repo tree) |
|---|---:|---:|---:|---:|---|
| Archaeology | 12 | 71 | 5 | 7.5 MB | csv, xlsx |
| Astronomy | 12 | 68 | 1,556 | 486 MB | mostly csv; also sp3, HDR, txt, tle, cdf, npz |
| Biomedical | 9 | 38 | 7 | 175 MB | xlsx, csv |
| Environment | 20 | 148 | 37 | 31 MB | csv, txt |
| Legal | 30 | 188 | 136 | 1.3 MB | csv, one html |
| Wildfire | 21 | 120 | 23 | 1 GB | csv, gpkg, xlsx, json |
| **Total** | **104** | **633** | **1,764** | **1.7 GB** | |

Sub-task counts are upstream's README figures too. At the pinned commit the
workload files hold 631: legal has 186, not 188.

The file counts and sizes above are upstream's README figures. The pinned
commit `b2e0d77` (fetched 2026-10-06) has 1,742 files and 666.9 MB under
`data/`: archeology (spelled so on disk) 5, astronomy 1,538, biomedical 8,
environment 37, legal 132, wildfire 22 (32 MB, not 1 GB). Two tasks name
files the repository lacks (wildfire-hard-19, wildfire-hard-21), and 9
other `data_sources` entries do not resolve case-sensitively; see
[D0 progress](implementation-plan.md#d0-progress). Use the measured counts.

- **Answers are public and in plain text** in `workload/<domain>.json`. Each
  task has `query`, `answer`, and `answer_type`. Most answer types are
  `numeric_exact`; the rest are string or list types, exact or approximate.
  Only some upstream scoring paths are deterministic; see the scoring policy
  below (source checked 2026-10-04).
- **Discovery has ground truth.** Each task's `data_sources` field lists the
  files it needs (some as globs). Each task also has a `deepresearch_subset`
  field: the needed files mixed with distractors.
- **Intermediate steps can be evaluated.** The 631 sub-tasks can be run as
  separate prompted tasks. This differs from evaluating intermediate values
  in one end-to-end program; upstream pipeline evaluation uses a model.
- **Discovery is trivial in some domains.** Archaeology has 5 files and
  biomedical has 7, while astronomy has 1,556 and legal 136. Report discovery
  per domain, and add a pooled condition where all 1,742 files form one
  collection. The pooled condition is our variant, so report it separately.
- **The licence is unclear.** The GitHub repo has no licence file, and the
  [Hugging Face copy](https://huggingface.co/datasets/eugenie-y/KramaBench)
  says only `license: cc`. Use the data locally. Never commit data, answers,
  or traces that quote answers, and don't publish them.
- **Possible contamination.** The answers have been public since 2025, so the
  model may have seen them in training. The no-tools baseline is a probe;
  it cannot distinguish memorisation, prior knowledge, and guessing, or prove
  the absence of contamination.
- **Reference scores** from the repo's leaderboard, on older hosted models: the
  best overall is 22.08% (DS-GURU self-correcting with o3); asking the model
  directly with no pipeline gets at most 9.64%. These are context, not targets,
  and they come from a different model class than our local 27B model.
  They are not the latest paper results: the
  [March 2026 paper revision](https://arxiv.org/abs/2506.06541v3) reports 55%
  end-to-end accuracy and 62% with perfect retrieval for its best systems.
  Pin both metric definitions and experimental conditions before comparing
  any of these numbers with this project's results.

### DiscoveryBench is optional (D7)

Source: [allenai/discoverybench](https://github.com/allenai/discoverybench),
data licensed ODC-By. It has 264 real and 903 synthetic tasks
([paper](https://arxiv.org/abs/2407.01725)). Its scorer compares the agent's
hypothesis with the gold one using an OpenAI chat model (`eval/eval.py` calls
`run_chatgpt_query_multi_turn`). That makes scoring dependent on the judge, so
it is not a primary benchmark here.

## Scoring policy

The inspected upstream [answer-type mapping](https://github.com/mitdbg/KramaBench/blob/main/benchmark/fixtures/answer_type_fixtures.json)
selects an LLM paraphrase judge for approximate strings. Its
[metrics implementation](https://github.com/mitdbg/KramaBench/blob/main/benchmark/metrics.py)
also uses a hosted judge for approximate string elements in lists, and the
[evaluator](https://github.com/mitdbg/KramaBench/blob/main/benchmark/evaluator.py)
uses a hosted model for pipeline evaluation. Disabling pipeline evaluation
alone does not remove all judge calls. These links track upstream; D1 must
record and verify the exact commit, fixture mappings, and selected code paths.

The default is a **local deterministic variant**, covering all answer types:

| Answer type | Default answer score |
|---|---|
| Numeric exact, string exact | Pinned upstream `success` comparison |
| Numeric approximate | Pinned upstream `rae_score` |
| List exact | Pinned upstream `f1` |
| String approximate | Deterministic `success` string comparison; no semantic judge |
| List approximate | Deterministic `f1`; no semantic judge |

The last two rows intentionally penalise paraphrases and are deviations from
upstream. Disable pipeline judging. Pin normalisation, numerical tolerances,
zero-target handling, list parsing, and aggregation in D1, with synthetic edge
cases. Never evaluate model output as host Python while parsing an answer;
use validated JSON or a safe literal parser. Document any resulting upstream
deviation. Equal-weight task means are the default answer-score aggregate;
report per-type and per-domain results as well.

**As implemented in D1** (`eval/kramabench/scoring.py`, profile
`local-deterministic-v1`). The comparators are a port of upstream
`benchmark/metrics.py` at `b2e0d77`; the file and the answer-type fixture
mapping are pinned by SHA-256. An offline test loads the upstream module
itself, with its LLM and NLP imports stubbed, and checks that the port
agrees on a grid of synthetic predictions and targets. It runs whenever
the evaluator store is present. Upstream cannot be imported directly:
`metrics.py` imports its OpenAI client at load, and `F1` calls `eval()` on
model output. Pinned behaviour, kept as upstream has it: `success` lowercases
and strips strings, compares numbers (including numeric strings) exactly, and
accepts a float within 1e-6 relative, 1e-6 absolute, or 1e-4 of the larger
magnitude; for a zero target it requires |p| < 1e-6. `try_convert_to_number`
strips `%` without dividing, while `rae_score` and list parsing divide by 100.
Booleans count as integers. `rae_score` is 1 / (1 + |p - t| / |t|) and 0 for
a zero target. List elements match strings case-insensitively and numbers
within 1e-6 relative (`f1`) or 1% (`f1_approximate`). Deviations: the two
judge rows in the table above; `ast.literal_eval` in place of `eval()` (the
same result for literals, and expressions are never executed); a NaN
`rae_score` becomes 0; and every task in the frozen manifest gets a score row,
whereas upstream `evaluate_results` silently drops tasks whose evaluation
raises. Every gold answer in the 104 tasks and both smoke tasks scores 1.0
and strict under this profile (offline test).

Report the continuous answer score separately from **strict answer accuracy**:
exact types must pass the pinned comparator, list types require F1 = 1, and
numeric-approximate answers must meet an absolute/relative tolerance frozen
in D1 before model evaluation. Approximate strings use the deterministic
comparison in this variant. Do not call a mean partial-credit score accuracy.
Frozen in D1 before any model run: `numeric_approximate` is strict when
|p - t| <= max(1e-6, 0.01 |t|). The 1% is upstream's own list-element
tolerance in `F1Approximate`.

An upstream-compatible judged comparison is optional, requires a declared
spend cap, and runs separately. Pin judge, prompts, settings, aggregation,
dataset revision, and task condition; record judge responses and failures.
Replacing a hosted judge with a local model is another scoring variant;
upstream ships `benchmark/llm_tools/ollama_interface.py`, a possible starting
point.
Replaying cached judge outputs is deterministic; fresh judging need not be.
Neither the local variant nor verified success is directly comparable to a
published upstream headline score.

## Three conditions on one benchmark

| Condition | What the agent gets | What it measures |
|---|---|---|
| No tools | The question only | Prior knowledge and a contamination probe |
| Given files | Question and labelled `data_sources`; only those raw files mounted | Analysis with oracle file selection |
| End to end | Question and scoped catalogue search; only selected, discovered files mounted | Catalogue discovery plus analysis |

Use the same tasks, scorer, analysis loop, and declared analysis budgets for
given-files and end-to-end comparisons; report discovery time and its separate
budget. Their difference measures the effect of the discovery condition,
including its context and tool overhead. No-tools versus given-files also
changes execution and interaction, so neither gap is a pure causal attribution.

Whether the agent sees each task's `answer_type` is decided in D1 and applied
identically in all three conditions. It is a format specification, not gold,
but it changes the task, so record it in every run and report it alongside
any comparison with upstream numbers. **Decided: hidden**
(`eval.answer_type_visible: false`); see
[D1 progress](implementation-plan.md#d1-progress).

The end-to-end policy is catalogue-mediated access: IDs must first be returned
by search, then explicitly selected for a `run_python` call. The application
enforces the manifest mapping and file mounts. Filesystem enumeration cannot
reveal the rest of the collection. A full-collection filesystem agent would
be a separate, labelled condition. Retrieval metrics rank unique files after
deduplicating card chunks, and report whether required file sets exceed k.
A retrieved group card counts as retrieving its listed members; report
file-level and group-level recall separately so groups cannot hide misses.

Gold answers, reference solutions, and benchmark sub-task instructions stay
outside prompts, tools, indexes, and sandbox mounts. Only the given-files
condition receives oracle `data_sources`. Discovery scoring reads those labels
in the evaluator. Gold sub-task prompting is a separate assisted diagnostic;
model-generated decomposition is allowed in the main condition.

## Specialization comparison and practical acceptance

The three benchmark conditions diagnose this system; they do not establish
that its custom loop is preferable to an existing agent. Add a D3 development
comparison with a minimal general code-execution baseline: a question, labelled
files, a generic analysis prompt, bounded Python execution and error feedback,
and a final answer/program submission. It has no catalogue search or specialized
re-plan policy. Freeze its prompt, tool contract, and retry policy before the
paired run. It must be a credible execution baseline, not an intentionally
underpowered single attempt.

Use the same local model/settings, selected-file mounts, sandbox packages,
answer visibility, scorer, and comparable wall/output/step budgets. Give both
conditions the same required answer/program output contract and independently
apply the same verifier to submitted programs. Record other differences rather
than treating the comparison as a clean causal attribution. Share the trusted
sandbox and evaluator where possible; this comparison tests loop/prompt
specialization, not the need for the sandbox infrastructure itself. Reusing an
existing compatible runner is preferred to building a second full agent. If
only a thin local runner is feasible, label it a proxy and bound the claims;
these results cannot establish superiority to ChatGPT or Codex.

Before the paired development comparison, record numeric per-task and total
run time limits, repeat count, the minimum worthwhile strict-accuracy gain,
and the acceptable correctness/runtime tradeoff. Choose them from the intended
local research use and available compute, not comparative outcome scores.
Measure median and tail latency, all-task budget-failure rate, verifier time,
and operator minutes spent preparing inputs, intervening, or reviewing each
condition. Keep manual answer repairs outside autonomous scores and disclose
all interventions; distinguish one-time setup from per-task effort.

Report paired correctness differences with uncertainty and audit coverage
separately. At D3, record continue, simplify, stop, or inconclusive against the
registered criteria before catalogue expansion. At D4, apply the registered
discovery/runtime criteria to end-to-end versus given-files results. Audit
coverage can justify infrastructure for a stated requirement even without an
accuracy gain, but does not establish adoption or the necessity of a custom
loop. A negative or inconclusive finding remains a valid research result.

## Metrics

| Layer | Measurements |
|---|---|
| Discovery | File recall@k, complete-set recall, per domain and pooled; latency |
| Answers | Local-variant answer score and strict answer accuracy, per type/domain and overall, with uncertainty |
| Sub-tasks | Separately prompted sub-task scores, labelled as assisted diagnostics; uncertainty clustered by parent task |
| Provenance | Reproduction and observed-access verification rates among submitted programs; counts and all-task verified success |
| Calibration | Coverage versus error with abstention; reliability by confidence bin, with sample counts |
| Operations | Wall-clock time, prompt tokens evaluated versus cached, thinking and output tokens, steps, sandbox runs, tool-call repairs, and errors per task |
| Failure analysis | Counts per failure-taxonomy category among failed runs, per condition and domain |
| Sandbox safety | Blocked network, blocked writes outside scratch, killed runaway processes |

**Verified success** requires a strictly correct answer, complete trusted
data-file read records with matching hashes, and a successful fresh rerun under
the [provenance contract](data-and-provenance.md#provenance-contract), including
nonempty observed inputs and reconciliation with claimed files.
Failure to reproduce fails verified success but does not overwrite the raw
answer score. Reproduction alone does not establish correctness or causal use
of the observed inputs. No-tools and inlined-data baselines have no program,
so their provenance metrics are not applicable.

Timeouts, crashes, abstentions, and malformed outputs receive zero in all-task
answer and verified-success aggregates. Reconcile every run against the frozen
task manifest so missing upstream evaluator rows cannot silently shrink the
denominator. Report conditional reproduction rates with submitted/total counts;
do not substitute them for all-task success. Coverage curves may additionally
report error among answered tasks, with coverage explicit.

## Splits and protocol

- Split tasks into development and holdout, stratified by domain, with a
  recorded seed. Keep parent tasks and their sub-tasks together, and assign
  smoke-test parents to development. D4 and D5 tuning use development only.
  Freeze prompts, configuration, thresholds, scorer, repeat count, and planned
  comparisons before D5's final holdout evaluation. Optional calibration is
  fitted within development before that freeze. Do not adapt using holdout
  results; later changes need a new untouched confirmation set or an explicit
  exploratory label.
- Run a fixed baseline and each candidate on the same tasks and report paired
  differences with uncertainty. With 104 tasks, small gains won't be
  significant. Sub-tasks offer diagnostic detail, not 631 independent samples.
  Resample paired parent tasks (preserving domain strata) and keep their
  sub-tasks together; state the uncertainty method and sample counts.
- Repeat model-dependent runs to estimate run-to-run variance.
- Record the repository revision, KramaBench commit, catalogue and index
  manifest, model name and settings, Ollama version, tool schemas, budgets,
  raw outputs, programs, failures, and timings for every run.

## Offline versus live checks

Default checks are offline and deterministic: card generation, retrieval-only
discovery scores against prepared local indexes, sandbox isolation tests, the
local deterministic scorer, the verifier, and recorded-response replays. Runs
that call the local generative model are separate commands. They make no paid
calls but take time, so keep them out of the default check run. Hosted judging
is a third, explicitly opt-in path; offline checks must work without credentials
and must not initiate network requests.
