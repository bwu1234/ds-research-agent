# Changes needed in rag-toolkit

Baseline: source inspected at `fa2a92a8929ddd47fcb4f9ec9fa603e0faef1eae` on
2026-09-29. The local checkout was at `3f56da8` on 2026-10-03. These are
proposed upstream work items, not applied changes. Recheck status against the
pinned revision before implementation.

## Existing capabilities to reuse

Markdown front matter, configurable carried metadata and headers, named
corpora, a separate index directory, hybrid retrieval, reranking, metadata
filters, incremental indexing, and read-only MCP search already exist. Dataset
cards are Markdown, so the catalogue fits the existing ingestion path.

## Configuration and integration work

This belongs in this repository and needs no upstream code:

- Generate the dataset cards with stable paths and `domain`, `catalogue_id`,
  `file_path`, and `format` metadata.
- Generate a dedicated configuration and index directory. Replace the default
  header, configure carried metadata, and restrict the served registry to the
  card corpus.
- Pin a toolkit revision and demonstrate MCP client compatibility.
- Maintain the manifest mapping from returned document IDs to catalogue IDs.
- Run indexing as a separate job and restart the service after publication.
- Compute discovery metrics in this repository. Use the toolkit's retrieval
  evaluator only where its schema fits.

## Proposed upstream work

| ID | Change | Need / timing | Workaround | Acceptance |
|---|---|---|---|---|
| R1 | Allowlisted structured metadata in MCP result entries | Return `catalogue_id` and `file_hash` directly; D0–D2 | Join the manifest; render key fields in the chunk header | Metadata matches the indexed card; an allowlist prevents accidental exposure; existing clients tolerate the extra fields |
| R2 | Index generation diagnostics and deliberate cache refresh | Record the generation each search run used, including optional D2 card ablations | Stop, index, restart; track the generation outside the toolkit | Search reports a stable generation; a failed publication keeps the old one |
| R3 | Invalidate chunks when carried metadata or header values change | Optional card-format ablations in D2 | Full reset and reindex per profiler version | A card with an unchanged body but changed metadata is updated in both indexes |
| R6 | MCP client compatibility fix, if D0 fails | Conditional D0 blocker | None; keep the D0 gate open | A pinned client passes discovery, list, and call on the real server |
| R7 | Distinguish an unbuilt index from zero filter matches | Avoid misreading an empty result in D0–D2 | Check the manifest and index status in this repo | Empty index, no filter matches, and low-score rejection are distinguishable |

R4 (service authentication) and R5 (full-document reads) are not needed: the
project runs locally for one user, and the sandbox reads raw files directly.

Open focused upstream issues or PRs only when a milestone shows the need.
Keep generic contracts and tests upstream, and benchmark-specific code here.
