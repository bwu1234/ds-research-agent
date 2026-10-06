import pytest

from ds_research_agent.catalogue import CatalogueEntry, Manifest
from ds_research_agent.retrieval import RetrievalError, SearchResponse, resolve


def entry(path: str) -> CatalogueEntry:
    return CatalogueEntry(
        catalogue_id=f"d/{path}",
        domain="d",
        file_path=path,
        size=1,
        file_hash="sha256:0",
        format="csv",
        document_id=f"{path}.md",
        card_hash="sha256:1",
        profiler_version=1,
        parse_status="ok",
    )


def response(*doc_ids: str) -> SearchResponse:
    return SearchResponse.model_validate(
        {
            "query": "q",
            "corpora": ["c"],
            "returned": len(doc_ids),
            "results": [
                {
                    "rank": i,
                    "score": 0.5,
                    "chunk_id": f"{d}::0",
                    "document_id": d,
                    "source": d,
                    "text": "t",
                }
                for i, d in enumerate(doc_ids, 1)
            ],
        }
    )


MANIFEST = Manifest(
    profiler_version=1, corpus="c", entries=[entry("d/a.csv"), entry("d/sub/a.csv")], skipped=[]
)


def test_resolves_by_exact_document_id() -> None:
    pairs = resolve(response("d/sub/a.csv.md", "d/a.csv.md"), MANIFEST)
    assert [e.file_path for _, e in pairs] == ["d/sub/a.csv", "d/a.csv"]


def test_unknown_document_is_an_error_not_a_basename_match() -> None:
    with pytest.raises(RetrievalError, match="not in the manifest"):
        resolve(response("other/a.csv.md"), MANIFEST)
