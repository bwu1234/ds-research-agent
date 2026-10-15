"""Resolve search results to catalogue entries through the manifest."""

from __future__ import annotations

from ds_research_agent.catalogue import CatalogueEntry, CatalogueGroup, Manifest
from ds_research_agent.retrieval.mcp_adapter import Passage, RetrievalError, SearchResponse


def resolve(
    response: SearchResponse, manifest: Manifest
) -> list[tuple[Passage, CatalogueEntry | CatalogueGroup]]:
    """Pair each passage with its file entry or group by exact document ID.

    An unknown document ID means the index and manifest are out of step; that
    is an error, never a basename guess.
    """
    by_doc: dict[str, CatalogueEntry | CatalogueGroup] = {
        **manifest.by_document_id(),
        **manifest.groups_by_document_id(),
    }
    out: list[tuple[Passage, CatalogueEntry | CatalogueGroup]] = []
    for p in response.results:
        record = by_doc.get(p.document_id)
        if record is None:
            raise RetrievalError("response", f"document {p.document_id!r} is not in the manifest")
        out.append((p, record))
    return out
