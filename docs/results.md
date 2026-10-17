# Results

Current results, kept up to date as comparisons land. The dated evidence
behind each figure (batches, commits, replay checks, decisions) is in
[implementation-plan.md](implementation-plan.md): D1 progress for the
baselines, D3 progress (g) to (n) for the agent comparison, D2 progress
(a) to (e) for discovery, and D4 progress (c) for end to end. When a new
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

## Discovery (D2)

All 53 development tasks, domain-filtered search, catalogue v3m3 (profiler
v3 cards; group cards for families of three or more). Gold sets are the
tasks' resolved `data_sources`. No answers are computed here. Dated evidence:
implementation plan, D2 progress (a) to (e).

| Method | Recall | Complete set | Notes |
|---|---|---|---|
| Single-shot rag hybrid, v2 cards, top 10 | 0.76 | 0.68 | (a) |
| Single-shot rag hybrid + collapse, v3m3, top 10 | 0.91 | 0.75 | ten cards shown, nothing selected |
| Agentic discovery (`discovery_agent`), v3m3 | 0.79 | 0.70 | median 2 files chosen, precision 0.79; median 108 s per task |

Registered comparison, agent minus single-shot top 10 on the same catalogue:
complete set −0.057 (−0.208 to +0.075), **inconclusive**; recall −0.116
(−0.206 to −0.020), though the single-shot figure counts reachable cards, not
a choice. Group cards plus collapse are what moved single-shot discovery:
against v2, recall@10 +0.145 (+0.065 to +0.226). Astronomy stays weakest
(agent complete set 0.33; single-shot 0.17). Rag-toolkit's pooled
(unfiltered) search is not deterministic across runs; the domain-filtered
search used here is.

## End to end (D4), Ollama 0.40.4

The `end_to_end` condition: one single-shot search (collapsed, top 10) seeds
the files the agent may use; it reads cards and selects files by catalogue ID
inside `run_python`, and only selected files are mounted. Everything else
matches `given_files`, which ran beside it. Two passes per condition at one
commit, pooled per task; criteria registered in plan entry D4 (a) before any
end-to-end run.

| | `given_files` | `end_to_end` | Difference (95% interval) |
|---|---|---|---|
| Strict accuracy, 41 primary | 54.9% | 40.2% | **−14.6** (−28.1 to −1.2) |
| Strict accuracy, all 53 | 55.7% | 42.5% | −13.2 (−26.4 to −0.9) |
| Answer score, 41 primary | 0.610 | 0.480 | −13.0 (−26.1 to −0.3) |
| Median time per task, 41 primary | 2.1 min | 4.2 min | 2.0× |
| Verified success, all 53, per pass | 35.8%, 39.6% | 34.0%, 32.1% | — |

Registered rule: retained if the lower bound is above −10 and median time is
at most 1.25×; worse if the upper bound is below −10. Decision:
**inconclusive** under that rule. The whole interval is below zero, so end to
end is **worse than given files**; what is not shown is a loss larger than 10
points. It also takes twice as long, so it could not have been retained at
any accuracy.

The seed search contained every needed file for 75% of tasks, but the
agent's final selection was complete for 55% (recall 0.72, precision 0.70).
Splitting each end-to-end run by how far its files got, and pairing it with
`given_files` on the same task (41 primary, both passes; diagnostic, not
part of the registered rule):

| Files in the run | Runs | `end_to_end` strict | `given_files`, same tasks | Difference (95% interval) | Share of the −14.6 |
|---|---|---|---|---|---|
| Every needed file selected | 42 | 57.1% | 66.7% | −9.5 (−21.4 to 0.0) | −4.9 |
| Search found them, selection incomplete | 18 | 33.3% | 50.0% | −16.7 (−44.4 to +11.1) | −3.7 |
| Search missed a needed file | 22 | 13.6% | 36.4% | −22.7 (−40.9 to 0.0) | −6.1 |

File choice among found files is about a quarter of the loss. Runs that
selected every needed file still trail given files on the same tasks, so the
discovery steps also cost the analysis; tasks the search misses are the
hardest ones for given files too. Each state holds only 9 to 21 parent
tasks, so these intervals are wide.

End-to-end runs use more steps (median 14–16 model requests against 8–9)
and hit the step budget more often (7–8 of 53 against 2–4). Verified success
is close between conditions: when end-to-end runs submit, their file access
checks out as often. `given_files` itself moved from 60.4% to 50.9% strict
(all 53) between its two passes on the same setup.

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
- The D1 to D3 analysis results use given files. End to end (D4) has one
  configuration: single-shot seed search, 10 cards, the given-files step
  budget. Its failed runs are not yet labelled, and accuracy has not been
  split by whether the selection was complete.
- These figures describe this agent with this local model. They are not
  comparable to published KramaBench scores of other systems, which use
  different models and judged scoring.
