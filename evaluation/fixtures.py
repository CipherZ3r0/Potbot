"""
Deterministic synthetic fixtures for retrieval evaluation tests.

Constructs ``SearchResult`` dataclass objects (the type actually returned by
the retrieval strategies) plus dict-style results (used by legacy callers)
with fully controlled positions for the expected chunk_id.
"""

from domain.models import SearchResult


def make_search_result(chunk_id: str, rank: int = 0, score: float = 0.0) -> SearchResult:
    """Create a synthetic SearchResult with controlled identity."""
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"synthetic chunk text for {chunk_id}",
        file_name="fixture.txt",
        source_file="fixture.txt",
        file_type=".txt",
        page_number=1,
        score=score,
    )


def ranked_results(total: int = 10, relevant_chunk_ids=("expected",)) -> list:
    """
    Build a ranked result list of length ``total``.

    The chunk from ``relevant_chunk_ids`` is placed at the position implied by
    ``total``/indexing by its requested rank (1-based). ``relevant_chunk_ids``
    may be a single id or an iterable paired with explicit ranks.
    """
    results = [make_search_result(f"candidate_{i}", rank=i, score=10.0 - i) for i in range(total)]
    return results


def results_with_expected_at(rank: int, total: int = 10) -> list:
    """
    Return a result list where chunk_id ``"expected"`` sits at 1-based ``rank``.

    Rank is 1-based: rank=1 puts the expected chunk first (best MRR=1.0).
    """
    results = []
    for i in range(1, total + 1):
        if i == rank:
            results.append(make_search_result("expected", rank=i, score=10.0 - i))
        else:
            results.append(make_search_result(f"candidate_{i}", rank=i, score=10.0 - i))
    return results


def dict_results_with_expected_at(rank: int, total: int = 10) -> list:
    """Dict-style mirror of :func:`results_with_expected_at` (legacy callers)."""
    return [
        {"chunk_id": "expected", "score": 10.0 - rank}
        if i == rank
        else {"chunk_id": f"candidate_{i}", "score": 10.0 - i}
        for i in range(1, total + 1)
    ]


def no_hit_results(total: int = 10) -> list:
    """A full result list that never contains the expected chunk_id."""
    return [make_search_result(f"candidate_{i}", rank=i, score=10.0 - i) for i in range(1, total + 1)]