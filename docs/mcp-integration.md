# MCP integration plan

## Inspected baseline

Inspected 2026-09-29 against rag-toolkit commit
`fa2a92a8929ddd47fcb4f9ec9fa603e0faef1eae`.
This is source inspection, not a completed client/server compatibility test.

- [Server contract](https://github.com/bwu1234/rag-toolkit/blob/fa2a92a8929ddd47fcb4f9ec9fa603e0faef1eae/docs/mcp-server.md)
- [Tool definitions and result serialization](https://github.com/bwu1234/rag-toolkit/blob/fa2a92a8929ddd47fcb4f9ec9fa603e0faef1eae/rag/tools.py)
- [Configuration path resolution](https://github.com/bwu1234/rag-toolkit/blob/fa2a92a8929ddd47fcb4f9ec9fa603e0faef1eae/rag/config/settings.py)

Pin this dependency during D0; review contract changes before upgrading. The
local checkout was at `3f56da8` on 2026-10-03, so recheck this contract against
the revision you pin.

## Existing tools

`rag_list_corpora` lists configured corpora and index information.
`rag_search` returns passages rather than a generated answer. Its inputs are:

| Argument | Current contract |
|---|---|
| `query` | Required string |
| `corpus` | Name or list of names; multiple names require the corresponding pooled index |
| `top_k` | Number of returned passages, 1–20 |
| `max_chars` | Positive per-passage character budget; default 1200 |
| `filters` | `equals`, `any_of`, and integer/date `range`; all conditions must hold |

Example tool arguments after indexing the dataset cards:

```json
{
  "query": "identity theft reports by metropolitan area",
  "corpus": "kramabench-cards",
  "top_k": 10,
  "max_chars": 2400,
  "filters": {
    "equals": {"domain": "legal"}
  }
}
```

Filters work only for `document_id` and configured carried metadata. An invalid
field is an error. The application fixes the corpus, and in per-domain
conditions the domain filter, before emitting the MCP call. The agent cannot
remove those filters. The underlying service does not enforce user identity.

The payload contains `query`, `corpora`, `pooled`, `candidate_count`,
`returned`, and `results`, plus some diagnostics. Results carry `rank`, `score`,
`chunk_id`, `document_id`, `source`, and `text`; optional fields include `page`,
`header`, `context`, `truncated`, and `full_length`.

Arbitrary metadata such as `catalogue_id`, `file_path`, and `file_hash`
is not currently serialized into each result. Resolve it using the export
manifest/source store and include key provenance in configured headers.
`score` is a ranking signal, not a truth/confidence probability.

Empty filtered retrieval can occur even with a populated index. Use explicit
index status and the ingestion manifest to diagnose it; the current empty
result hint alone cannot prove the index is unbuilt or the fact does not exist.

## Compatibility gate

The inspected server explicitly accepts only protocol revision `2026-07-28`.
Its documentation describes `server/discover` and per-request version and
capability metadata, and rejects a legacy `initialize` flow. These are facts
about this checkout, not a claim that every MCP client supports that revision.

D0 must select and pin a compatible client by exercising discovery,
`tools/list`, and `tools/call` against the actual server. If available client
libraries cannot connect, open a scoped upstream compatibility decision and
record it. Do not silently substitute an assumed handshake, build an untested
protocol shim, or report connectivity from a subprocess launch alone.

Use a small synthetic card collection for D0's filtered-search and identity
mapping checks. Expand this setup to the complete benchmark catalogue in D2;
full catalogue coverage does not block D3's first given-files workflow.

## Local configuration recipe

Use the rag-toolkit virtual environment's interpreter and set its working
directory explicitly. Keep its dependencies separate from this project's.

The following is a template to instantiate during D0, not a working checked-in
configuration. Replace every absolute placeholder. `base` is resolved relative
to the overlay file; other relative corpus/index paths are resolved against
rag-toolkit's repository root. Absolute paths avoid that ambiguity.

```yaml
base: /ABS/PATH/rag-toolkit/rag/config/config.yaml
paths:
  corpus_dir: /ABS/PATH/ds-research-agent/data/cards/kramabench
  index_dir: /ABS/PATH/ds-research-agent/data/index
corpora:
  active: [kramabench-cards]
  registry:
    kramabench-cards:
      documents_dir: /ABS/PATH/ds-research-agent/data/cards/kramabench
      description: KramaBench dataset cards
chunking:
  carry_metadata: [title, domain, catalogue_id, file_path, format, card_kind, group_id]
  header:
    template: "{domain} | {format} | {file_path}"
retrieval:
  web_search:
    enabled: false
observability:
  turn_log:
    provider: none
```

Base/overlay registry mappings merge, so other base corpora remain listed.
`active: [kramabench-cards]` is a selection default, not a discovery or access boundary.
D0's generated configuration should flatten the inherited configuration and
restrict the registry to the card corpus before it is served. Do not
expose a shared service with unrelated base corpora available.

`card_kind` (`file` or `group`) and `group_id` support group cards (see
[data-and-provenance.md](data-and-provenance.md#group-cards)). Group cards
have no single `file_path`; the profiler writes the group directory there so
the header template stays valid.

The base configuration embeds queries through the same Ollama daemon as the
agent's model (`embedding.provider: ollama`, `qwen3-embedding:0.6b`). D0
measures whether a search between model turns evicts the agent's cached prompt
prefix. The served overlay must also enable no retrieval step that calls a
generative model (for example HyDE query expansion), which would load a second
LLM next to the agent's.

Keep generic retrieval defaults initially. The baseline's EDGAR-specific
header must be overridden. Measure any changes to chunking, reranking, or
generation on the D2 discovery metrics before adopting them.

Once the overlay and exports exist, these existing toolkit commands are the
intended integration recipe; they have not been run for this new project:

```bash
cd /ABS/PATH/rag-toolkit
.venv/bin/python -m rag.cli --config /ABS/PATH/ds-research-agent/config/local.rag.yaml index --corpus kramabench-cards
.venv/bin/python -m rag.cli --config /ABS/PATH/ds-research-agent/config/local.rag.yaml index-report --corpus kramabench-cards
.venv/bin/python -m rag.mcp --config /ABS/PATH/ds-research-agent/config/local.rag.yaml
```

Use `index --reset` when changing embedding/header/carried-metadata settings
as required by the toolkit's manifest. D0 owns validating this setup with the
actual selected models and files; model downloads and live calls are separate
from documentation creation.

## Adapter behavior

- Maintain one service process per active configuration/generation. Drain
  stderr without mixing logs into stdout, which is the protocol channel.
- Validate tool schemas and response payloads; reject invalid evidence IDs and
  unexpected shapes. Accept supported additive fields without losing evidence.
- Distinguish startup, transport, protocol, argument, index, and search errors.
  Allow bounded retries only where appropriate for read operations.
- Enforce per-call deadlines, cancellation/process cleanup, passage budgets,
  and bounded result sizes; include cold-start time in diagnostics.
- Preserve exact excerpts and provenance. Wrap source content as untrusted
  data and keep it separate from instructions and tool permissions.
- Resolve a truncated card through the application's `read_card` tool, which
  reads the card from the manifest subject to the run's discovered-ID policy
  in [architecture.md](architecture.md). The current MCP server has no
  full-document read tool, and the sandbox reads raw files directly.
- Log the query, scope, returned evidence references, latency, and errors in
  this application's trace. The toolkit's chat turn logs are not MCP traces.

## Index refresh and later HTTP deployment

Indexing is never an agent tool. For the first prototype, stop the local
service, publish complete exports, index them, and restart. Retrievers are
cached in process; do not assume a running process automatically adopts every
sparse/dense/config change consistently.

For continuous operation, build versioned generations and switch only after
validation, preserving the generation used by each in-flight run. HTTP adds
service authentication, server-enforced authorization, scoped caches, limits,
and readiness requirements; the existing `/mcp` endpoint is not sufficient
authorization infrastructure by itself.
