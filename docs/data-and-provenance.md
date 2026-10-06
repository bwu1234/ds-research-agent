# Data, catalogue, and provenance

Status: the D0 file-card schema, manifest, and KramaBench fetch are implemented; other records are proposed.

## Data handling

Benchmark data lives in an ignored directory (`data/`), fetched by a script
pinned to a KramaBench commit, with checksums. Never commit data files,
benchmark answers, or run traces that quote answers. The KramaBench licence
is unclear (see [evaluation-plan.md](evaluation-plan.md)); generated cards,
indexes, answer-bearing artifacts, and traces also stay private. The write-up
reports aggregate measurements and methodology, with synthetic data for public
demos. Test fixtures in Git are small synthetic files made for this repo.

Data files are untrusted input. A file's contents can't change tool
permissions or the agent's instructions, and the sandbox only ever reads them.
Keep workload labels, reference solutions, and gold sub-task material in a
separate evaluator-only location, excluded from profiling, indexing, prompts,
and sandbox mounts. Fetching the benchmark does not expose its repository to
the agent.

The fetch script (`eval/kramabench/fetch.py`, implemented in D0) splits the
KramaBench repository by visibility into stores whose paths are set in the
`kramabench` configuration section and may not overlap:

| Upstream path | Destination |
|---|---|
| `data/` | Agent-visible raw-file store; the only source for profiling and the catalogue |
| `workload/` | Evaluator-only (queries, answers, `data_sources`, sub-tasks) |
| `solutions/` | Evaluator-only (reference code per task) |
| `dr-input/` | Not catalogued; per-task copies of `data/` files that would duplicate the pooled collection |
| `benchmark/` and every other top-level entry | Evaluator-only, for pinning the scorer |
| `.git`, the clone itself | Evaluator-only checkout; never mounted |

Checksums (`SHA256SUMS` per store) are generated at fetch time and kept with
the data, not committed: a list of benchmark file names and hashes is
benchmark-derived. The pinned commit and tree hash are the committed
identity; `verify` rehashes the stores.

Task queries reach the agent only through the harness, never through a mount.

## Dataset cards

The profiler writes one Markdown card per data file. It is deterministic and
uses no LLM, so the same files and profiler version always produce the same
cards.

```markdown
---
title: State MSA Identity Theft Data/Utah.csv
domain: legal
catalogue_id: legal/0f3a9c1e
file_path: legal/State MSA Identity Theft Data/Utah.csv
file_hash: sha256:...
format: csv
profiler_version: 1
---
# Utah.csv (legal)

Rows: 12. Columns: 3.

| Column | Type | Nulls | Example values |
|---|---|---|---|
| Metropolitan Area | string | 0 | "Salt Lake City, UT Metropolitan Statistical Area" |
| Reports | integer | 0 | 1203, 877 |
| ... | | | |

Sample rows: ...
```

The example is illustrative. A minimal schema is settled on synthetic fixtures
in D0 and expanded for benchmark formats in D2. Optional card-content ablations
use separate development generations. Formats the profiler can't parse
get a minimal card (path, size, format, hash) and an entry in the profiler
error list. Cards note which reader in the sandbox package set loads the format.

Profile large files from bounded samples so profiling time and memory stay
fixed; record the sampling rule in the card and profiler version.

### D0 card schema

Implemented in `ds_research_agent/catalogue/profiler.py`, `PROFILER_VERSION = 1`:

- **Front matter**, all values YAML strings so rag-toolkit's `equals` and
  `any_of` filters match them: `title` (the file path), `domain`,
  `catalogue_id`, `file_path`, `file_hash` (`sha256:`), `format` (lower-case
  extension), `size`, `card_kind` (`file`), `parse_status` (`ok`,
  `unsupported`, `error`), `profiler_version`.
- **Identity.** `domain` is the first path component under the data root.
  `catalogue_id` is `<domain>/<first 10 hex digits of sha256(file_path)>`;
  the build fails on a collision. The card is written at
  `<cards_dir>/<file_path>.md`, and rag-toolkit's Markdown loader uses that
  relative path as the document ID, so `document_id = file_path + ".md"`.
  The manifest records this, and the adapter resolves results by exact
  document ID (confirmed against the real loader in the D0 search test).
- **Body.** Path, format, size, sandbox reader, and encoding (UTF-8, then
  cp1252, then latin-1) and delimiter (`,` `;` tab `|`, chosen from the
  header line) for CSV. One table per CSV, JSON array of objects, or JSON
  object key holding an array of objects: row and column counts, then per
  column the inferred type (integer, float, boolean, ISO date, string), null
  count, numeric or date range, and up to `max_example_values` distinct
  examples (strings quoted), then the first `card_sample_rows` rows. Cells
  are whitespace-collapsed, clipped to `max_cell_chars`, and `|`-escaped.
- **Sampling rule.** Column statistics use the first `sample_rows` rows; rows
  are counted to the end of the file, and the card states which applied.
  D0 decodes each file whole to choose an encoding; D2 must stream large
  files.
- **Skipped paths** (hidden files, symlinks, files outside a domain
  directory, paths containing newlines) get no card and are listed in the
  manifest with a reason.

### Group cards

Some collections are families of similar files: astronomy has 1,556 files,
and KramaBench's `data_sources` uses globs. Per-file cards for such families
are near-identical, so retrieval cannot rank them, and selecting hundreds of
IDs one at a time is impractical. The profiler therefore also writes a group
card for each directory or shared-schema family:

- front matter with `card_kind: group`, `group_id`, `domain`, member count,
  and the shared format (file cards carry `card_kind: file` and their
  `group_id`, if any);
- the shared schema, with columns that differ across members flagged;
- the member list (path, size, hash), with naming patterns such as date
  ranges summarised when the list is long.

A discovered group ID can be passed to `read_card` and `run_python`, selecting
exactly its listed members. Grouping is deterministic. Its granularity
(directory, schema, or both) is settled on development tasks in D2.

## Durable records

| Record | Minimum fields |
|---|---|
| CatalogueEntry | catalogue ID, domain, file path, size, content hash, format, card hash, profiler version, parse status, group ID if any |
| CatalogueGroup | group ID, domain, grouping rule, member catalogue IDs, card hash, profiler version |
| CatalogueGeneration | generation ID, profiler version, entry and group counts, index directory, rag-toolkit revision, build time |
| Run | run ID, task ID, parent task ID if applicable, split, condition, configuration hash, model name and settings, Ollama version, catalogue generation, input manifest, access policy, `answer_type` visibility, scoring profile, start/end time, stop reason, failure category (assigned in review) |
| Step | run ID, sequence number, model turn or tool call, input, output, prompt tokens evaluated, prompt tokens cached, thinking tokens, output tokens, tool-call repair retries, duration |
| Program | run ID, step, kernel session ID and cell sequence, source text, sandbox image ID and package list hash, selected input and group IDs/hashes, exit status, stdout/stderr (bounded, with truncation flag), runner-observed data-file reads/hashes, audit mechanism/version and completeness |
| Answer | run ID, value, answer type, model-claimed input IDs, final program reference, abstained flag, confidence |
| Verification | answer ID, fresh rerun program reference and output, comparator version/tolerances, reproduced flag, observed-access verified flag, mismatch or audit failure detail |
| Score | run ID, scorer commit/profile, answer score, strict correctness, verified success (nullable for no-program baselines), sub-task scores, judge configuration/response references if opted in |

Use timezone-aware UTC timestamps. Store the records in SQLite and large
outputs as files under an ignored `artifacts/` directory, referenced by hash.
The D3 given-files workflow needs a raw-input manifest before the full D2 index
exists; its catalogue generation may be null, but paths and hashes are required.

Exploration cells in the persistent kernel are Program records for the ledger
and failure analysis. Only the submitted final program, rerun in a fresh
process, can support a provenance claim; state carried between cells cannot.

Failure categories come from a fixed taxonomy: discovery miss, parse error,
wrong filter or join, formatting, budget exhausted, tool-call failure, other.
Change the taxonomy only between protocol versions, never during holdout.

## Identity and the index

rag-toolkit derives document IDs from relative paths through its loader. D0
must establish the mapping from a card's relative path to the returned
document ID using the real loader and search contract, and record it in the
manifest. The adapter resolves search results to `catalogue_id` through the
manifest, not by matching basenames.

## Catalogue lifecycle

1. Fetch or update the data files and record their hashes.
2. Profile every file into a staging directory. Record parse failures.
3. Index the complete staged card set as a new generation and verify the entry
   count.
4. Publish the generation and restart the MCP service so the cached retrievers
   pick it up. Every run records the generation it used.

Changing the card format (a D2 ablation) means a new profiler version, a new
generation, and a full reindex. Keep earlier generations until their runs have
been scored.

## Provenance contract

Keep three claims distinct:

1. **Observed access:** the trusted runner records which raw data files the
   program read, with hashes matching the run's immutable input manifest and,
   when indexed, its catalogue generation. Model-declared files are claims,
   not observations. Mounted, selected, and actually read files are separate
   sets. Resolve reads to stable input identities, including child-process and
   native-library access; missing audit coverage fails this check.
2. **Reproduction:** the final self-contained program recreates its intermediate
   artifacts and structured answer in a fresh sandbox with empty scratch,
   selected raw inputs, and the pinned environment. Record its reads again.
   Freeze the output comparison rules and numerical tolerances before scoring.
3. **Correctness:** the separate evaluator compares the answer against gold
   under the declared scoring profile. Neither file reads nor a successful
   rerun establishes that the calculation answers the question.

A provenance-verified answer requires both observed-access verification and
reproduction. For these data-dependent tasks, an empty raw-input read set
cannot support a provenance claim; reading a file still does not prove its
contents caused the output. Complete access records must agree with the final
program's claimed inputs. A program that reads data and hard-codes its answer
can still evade this check, so do not claim semantic or causal verification.
Use synthetic wrong-filter, wrong-join, and hard-coded-output cases to test and
demonstrate these limits, alongside read-audit failure cases.

Verified success additionally requires strict answer correctness as defined
in [evaluation-plan.md](evaluation-plan.md). Failed provenance leaves the raw
answer score intact but fails verified success. No-program baselines have no
provenance status. Preserve separate failure reasons in the ledger.
