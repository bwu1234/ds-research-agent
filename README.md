# DS Research Agent

A data research agent that answers questions over messy data files by running
Python in a sandbox, with a local model (`qwen3.8:27b-mlx` through Ollama).
Each answer records the files the sandbox observed it reading, the program
that computed the value, and whether a fresh rerun reproduced it.

**Finding so far:** on KramaBench development tasks, letting the model run
code in a sandbox beats pasting the files into the prompt by **+22 points of
strict accuracy**. Whether this project's custom agent loop beats a general
code agent (smolagents `CodeAgent` on the same sandbox) is **not yet
settled**: the custom loop leads in every pass (+5 to +10 points), but the
intervals include zero.

## Status

| Milestone | State |
|---|---|
| D0 Scaffold: config, model client, MCP retrieval adapter, dataset-card profiler | Done |
| D1 Evaluation harness: deterministic scorer, frozen split, SQLite run ledger, replay, baselines | Done |
| D3 Sandbox and given-files analysis: restricted container, read audit, persistent kernel, agent loop, rerun verifier, comparator | In progress; paired comparison run and decided |
| D2 Catalogue expansion and discovery | Planned |
| D4 End to end: search the catalogue, then analyse, with provenance | Planned |
| D5 Ablations and sealed-holdout evaluation | Planned |

Today the agent works on **given files**: the task's input files are mounted
read-only in the sandbox. Finding the files by searching a catalogue
([rag-toolkit](https://github.com/bwu1234/rag-toolkit) over MCP) is built at
the adapter level but not yet part of the agent's evaluated runs.

## Results

KramaBench development split, 41 tasks not used for tuning, local model,
local deterministic scorer. Comparison criteria were registered before the
runs.

| Condition | What the model gets | Strict accuracy |
|---|---|---|
| Inlined files | File text pasted into the prompt | 36.6% |
| `codeagent` | smolagents `CodeAgent` in the sandbox | 53.7% |
| `given_files` | This project's agent loop in the sandbox | **58.5%** |

| Comparison | Difference (95% interval) | Decision |
|---|---|---|
| P1: `given_files` − inlined files | **+22.0** (+4.9 to +39.0) | Code execution pays |
| P2: `given_files` − `codeagent` | +4.9 (−9.8 to +19.5) | Inconclusive; +8.5 on a two-pass repeat |

Development tasks only: the holdout stays sealed until D5. Setup, all-task
figures, runtime limits, failure analysis, and limitations are in
[docs/results.md](docs/results.md).

## Example task

> Report the average number of reported identity thefts for all metropolitan
> areas larger than one million in population in 2023.

(Abridged from KramaBench's legal domain.) The agent inspects the files,
writes and runs analysis code in the sandbox, and submits an answer with the
program that produced it. A verifier reruns that program in a fresh sandbox
and checks the reads the runner observed. Answer correctness, observed file
access, and reproduction are scored separately: a reproducible result can
still be wrong.

## Evaluation

The main benchmark is [KramaBench](https://github.com/mitdbg/KramaBench): 104
tasks and 631 sub-tasks over 1,742 files in 6 domains (at the pinned commit),
scored under three conditions: no tools, given files, and end to end. Default
scoring is a documented local deterministic variant; upstream judged scoring
is opt-in. Every run is recorded in a SQLite ledger and replays from recorded
model responses with zero differences. See the
[evaluation plan](docs/evaluation-plan.md).

## Repository boundary

| Repository | Responsibility |
|---|---|
| This repository | Agent loop, model client, dataset-card profiler, sandbox, run ledger, verifier, evaluation harness |
| `rag-toolkit` | Generic ingestion, indexing, hybrid retrieval, reranking, filtering, MCP serving |

The agent calls `rag_search` directly through MCP and owns the reasoning loop.
Indexing is a separate job.

## Start here

1. [Results](docs/results.md): current figures, comparisons, limitations.
2. [Implementation roadmap](docs/implementation-plan.md): milestones, dated
   evidence, decisions.
3. [Architecture](docs/architecture.md): components, agent tools, sandbox
   requirements.
4. [Evaluation plan](docs/evaluation-plan.md): benchmarks, conditions,
   metrics, splits.
5. [Data, catalogue, and provenance](docs/data-and-provenance.md): dataset
   cards, run records, provenance contract.
6. [MCP integration](docs/mcp-integration.md): current service contract and
   configuration recipe.
7. [Changes needed in rag-toolkit](docs/rag-toolkit-changes.md).

## Next

D4 runs end to end: search the catalogue for the input files, then analyse
them, with provenance. D5 runs focused ablations and the sealed-holdout
evaluation, and the first release ends with a write-up and a synthetic
provenance demo.

See [CLAUDE.md](CLAUDE.md) for contributor guidance and commands, and
[AGENTS.md](AGENTS.md) for the Codex entry point.
