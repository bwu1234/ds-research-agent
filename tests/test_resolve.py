import pytest

from ds_research_agent.catalogue import CatalogueEntry, CatalogueGroup, Manifest
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


def test_resolves_group_cards_to_their_group() -> None:
    group = CatalogueGroup(
        group_id="d/g-1",
        domain="d",
        rules=("pattern",),
        member_ids=("d/d/a.csv", "d/d/sub/a.csv"),
        directory="d",
        format="csv",
        document_id="d/_groups/g-1.md",
        card_hash="sha256:2",
        profiler_version=3,
    )
    manifest = MANIFEST.model_copy(update={"groups": [group]})
    (_, g), (_, e) = resolve(response("d/_groups/g-1.md", "d/a.csv.md"), manifest)
    assert g == group and isinstance(e, CatalogueEntry)
    assert [m.file_path for m in manifest.members(group)] == ["d/a.csv", "d/sub/a.csv"]
