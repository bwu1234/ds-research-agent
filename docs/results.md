# Results

Current results, kept up to date as comparisons land. The dated evidence
behind each figure (batches, commits, replay checks, decisions) is in
[implementation-plan.md](implementation-plan.md): D1 progress for the
baselines and D3 progress (g) to (n) for the agent comparison. When a new
comparison is decided there, update this page and the README headline in the
same change.

All figures are on the KramaBench **development split**. The holdout stays
sealed until D5. No hosted model was called.

## Setup

- Model `qwen3.8:27b-mlx` through local Ollama, thinking off, temperature 0,
  seed 0, `answer_type` hidden.
- Scorer: `local-deterministic-v1`, the local deterministic variant in
  [evaluation-plan.md](evaluation-plan.md). Upstream judged scoring was not run.
- Agent conditions share the sandbox image, packages, labelled input files,
  scorer, and budgets: 20 steps, 8,192 output tokens per request, 4,000
  characters per tool output, 300 s per cell, 30 min per task. Hitting a
  budget scores 0. The sandbox allows 8 GiB; the measured peak per run is
  at most 300 MiB (median 81 MiB), so memory is not a binding limit here.
- Primary set: the **41 development tasks not used for tuning**. The other 12
  (the sample) were read next to their gold answers while tuning the agent,
  which biases comparisons toward `given_files`; they are reported alongside
  and do not decide.
- Comparison criteria (minimum worthwhile gain, practical limits, repeat
  rule) were registered before the runs, in plan entry D3 (g).

## Conditions, Ollama 0.35.1

| Condition | What the model gets | Strict accuracy, 41 primary | Strict accuracy, all 53 | Answer score, all 53 | Median time per task | Budget stops (of 53) |
|---|---|---|---|---|---|---|
| No tools | The question only | — | 3.8% | 0.090 | 1.2 min | 3 output cap, 1 malformed |
| Inlined files | File text pasted into the prompt | 36.6% | 35.8% | 0.414 | 4.7 min | 9 output cap |
| `codeagent` | smolagents `CodeAgent` in the sandbox | 53.7% | 54.7% | 0.595 | 2.8 min | 10 |
| `given_files` | This project's agent loop in the sandbox | **58.5%** | **58.5%** | **0.665** | 2.1 min | 1 |

No tools and inlined files are the D1 baselines (single request, their own
budgets). The no-tools baseline was not reported on the 41-task subset.

## Paired comparisons

95% percentile bootstrap intervals over parent tasks within domain. Decision
rule: a worthwhile gain is at least +10 points with the lower bound above 0;
a gain is ruled out if the upper bound is below +10; otherwise inconclusive.

| Comparison | Question | Strict difference, 41 primary | All 53 | Decision |
|---|---|---|---|---|
| P1: `given_files` − inlined files | Does code execution pay? | **+22.0** (+4.9 to +39.0) | +22.6 (+7.6 to +37.7) | **Continue** |
| P2: `given_files` − `codeagent` | Does the custom loop pay? | +4.9 (−9.8 to +19.5) | +3.8 (−9.4 to +17.0) | **Inconclusive** |

Secondary, 41 primary tasks: answer score P1 +22.8 (+8.9 to +36.6), P2 +8.9
(−7.7 to +25.5). Verified success (submitted, reproduced on rerun, and
observed file access passed): `given_files` 31.7%, `codeagent` 34.1%,
difference −2.4 (−19.5 to +14.6), so it gives no audit-value case for the
custom loop.

With 41 tasks, a true +10 point gain would usually come out inconclusive, as
stated before the runs. Inconclusive is not evidence of a tie.

## Repeat on Ollama 0.40.4-rc0

Two full development passes of both agent conditions, a new baseline not
pooled with the 0.35.1 runs. The second pass was triggered by the registered
run-to-run rule.

| | `given_files` strict, all 53 | `codeagent` strict, all 53 | P2 strict, 41 primary |
|---|---|---|---|
| Pass 1 | 58.5% | 49.1% | +9.8 (−4.9 to +24.4) |
| Pass 2 | 62.3% | 54.7% | — |
| Two-pass average | — | — | +8.5 (−3.7 to +20.7) |

P2 stays inconclusive with the registered extra budget spent. The point
estimate favours the custom loop in every comparison made. Same-server
pass-to-pass strict flips: `given_files` 4 of 53, `codeagent` 7 of 53.

## Practical limits

Registered limits for one researcher on one Mac: median task 5 min or less,
p90 20 min or less, budget stops on 15% of tasks or fewer, a development pass
8 h or less. On 0.35.1, all 53 tasks:

| Condition | Median | p90 | Budget stops | Pass time | Meets limits |
|---|---|---|---|---|---|
| `given_files` | 2.1 min | 8.6 min | 2% | 3.4 h | Yes |
| `codeagent` | 2.8 min | 11.1 min | 19% | 4.2 h | No (budget stops) |

## Where `given_files` fails

Hand-labelled failure taxonomy (`taxonomy-v1`) for the 22 failed
`given_files` runs on 0.35.1: wrong filter or join 11, formatting 4, other 6,
budget exhausted 1. Typical analysis errors are a skipped qualifier in the
question, a column or flag not inspected, or a tool output cut at its cap.
Not all failures are the agent's: two tasks name a file absent from the
benchmark repository, and three string answers cannot match under the local
string scorer. `codeagent` failures are labelled only where the stop reason
decides it (budget exhausted, by rule); the rest are not yet labelled.

## Limitations

- Development tasks only; no holdout result exists yet.
- One pass per condition on 0.35.1; run-to-run variance there is measured only
  by a 12-task probe. The 0.40.4-rc0 repeat has two passes.
- The comparator uses its own prompt, so P2 compares two bundles, not single
  features.
- Runtime figures include the 12 tuned tasks. The 0.35.1 `given_files` batch
  spans two commits, documented in plan entry D3 (j).
- A third of inlined-files runs hit the output cap, so that baseline is a
  floor for what the model could do with more tokens.
- Given files only: the agent has not yet searched the catalogue itself. D2
  measures retrieval-only discovery so far (implementation plan, D2 progress
  (a)); end-to-end discovery is D4.
- These figures describe this agent with this local model. They are not
  comparable to published KramaBench scores of other systems, which use
  different models and judged scoring.
