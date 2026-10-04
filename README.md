# DS Research Agent

A planned data research agent. Given a question and a collection of messy data
files, it finds the relevant files, analyses them by running Python in a
sandbox, and returns an answer with provenance: the files it read (with
hashes), the program that computed each value, and a rerun showing the value
reproduces. It uses [rag-toolkit](https://github.com/bwu1234/rag-toolkit) as
an MCP search service over dataset cards, and a local `qwen3.8:27b-mlx` model
through Ollama.

**Status: planning repository.** There is no application, dependency set,
benchmark download, or score yet. All milestones are pending. This is a
portfolio and learning project.

## Example task

> Report the average number of reported identity thefts for all metropolitan
> areas larger than one million in population in 2023.

(Abridged from KramaBench's legal domain.) The output should name the files
used, include the program that computed the value, and record whether a rerun
reproduced it.

## Evaluation

The main benchmark is [KramaBench](https://github.com/mitdbg/KramaBench): 104
tasks and 633 sub-tasks over 1,764 files in 6 domains. It is scored under
three conditions: no tools, given files, and end to end. See the
[evaluation plan](docs/evaluation-plan.md).

Default scoring uses a documented local deterministic variant; upstream
judged scoring is opt-in. Answer correctness, observed data-file access, and
reproduction are measured separately. A reproducible result can still be wrong.

## Repository boundary

| Repository | Responsibility |
|---|---|
| This repository | Agent loop, model client, dataset-card profiler, sandbox, run ledger, verifier, evaluation harness |
| `rag-toolkit` | Generic ingestion, indexing, hybrid retrieval, reranking, filtering, MCP serving |

The agent calls `rag_search` directly through MCP and owns the reasoning loop.
Indexing is a separate job.

## Start here

1. [Implementation roadmap](docs/implementation-plan.md): milestones, model and
   compute, acceptance checks.
2. [Architecture](docs/architecture.md): components, agent tools, sandbox
   requirements.
3. [Evaluation plan](docs/evaluation-plan.md): benchmarks, conditions,
   metrics, splits.
4. [Data, catalogue, and provenance](docs/data-and-provenance.md): dataset
   cards, run records, provenance contract.
5. [MCP integration](docs/mcp-integration.md): current service contract and
   configuration recipe.
6. [Changes needed in rag-toolkit](docs/rag-toolkit-changes.md).

## First implementation task

Complete D0: scaffold the project, pin rag-toolkit and an MCP client, fetch
KramaBench, and prove filtered search on a small synthetic card collection
plus one model tool-call round trip. Then establish D1's scoring and development
sample and build D3's given-files workflow: sandbox, structured answer, ledger,
observed file reads, and a fresh rerun. Expand the benchmark catalogue in D2.

The first release ends with development ablations, a frozen holdout evaluation,
and a write-up with a synthetic provenance demo. Calibration and additional
benchmarks are optional extensions.

See [CLAUDE.md](CLAUDE.md) for contributor guidance and [AGENTS.md](AGENTS.md)
for the Codex entry point.
