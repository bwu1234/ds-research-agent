"""Resolve search results to catalogue entries through the manifest."""

from __future__ import annotations

from ds_research_agent.catalogue import CatalogueEntry, Manifest
from ds_research_agent.retrieval.mcp_adapter import Passage, RetrievalError, SearchResponse


def resolve(response: SearchResponse, manifest: Manifest) -> list[tuple[Passage, CatalogueEntry]]:
    """Pair each passage with its catalogue entry by exact document ID.

    An unknown document ID means the index and manifest are out of step; that
    is an error, never a basename guess.
    """
    by_doc = manifest.by_document_id()
    out: list[tuple[Passage, CatalogueEntry]] = []
    for p in response.results:
        entry = by_doc.get(p.document_id)
        if entry is None:
            raise RetrievalError("response", f"document {p.document_id!r} is not in the manifest")
        out.append((p, entry))
    return out
