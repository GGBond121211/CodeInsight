"""可解释 BM25 检索测试。"""

import pytest

from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.bm25 import search_chunks_bm25


def _chunk(path: str, start_line: int, text: str) -> SourceChunk:
    return SourceChunk(path, start_line, start_line, text)


def test_rare_terms_rank_above_common_terms() -> None:
    target = _chunk("target.py", 1, "rareterm common common")
    common_only = _chunk("common.py", 1, "common common common")

    results = search_chunks_bm25("rareterm common", (common_only, target))

    assert results[0].chunk is target


def test_term_frequency_increases_score_with_saturation() -> None:
    one_hit = _chunk("one.py", 1, "rare " + "filler " * 19)
    many_hits = _chunk("many.py", 1, "rare " * 10 + "filler " * 10)

    results = search_chunks_bm25("rare", (one_hit, many_hits))

    assert results[1].score > 0.0
    assert results[0].score > results[1].score
    assert results[0].score < results[1].score * 10


def test_longer_documents_are_normalized() -> None:
    short = _chunk("short.py", 1, "needle filler")
    long = _chunk("long.py", 1, "needle " + "filler " * 30)

    results = search_chunks_bm25("needle", (long, short))

    assert results[0].chunk is short


def test_path_filename_identifier_and_code_field_bonuses_are_applied() -> None:
    code = _chunk("src/shop/pricing.py", 1, "def apply_discount(amount): return amount")
    document = _chunk("docs/pricing.md", 1, "def apply_discount(amount): return amount")
    unrelated_path = _chunk("src/shop/models.py", 1, "value = 1")

    code_results = search_chunks_bm25("apply_discount", (document, code))
    path_results = search_chunks_bm25("pricing", (unrelated_path, code))

    assert code_results[0].chunk is code
    assert path_results[0].chunk is code


def test_same_scores_sort_by_path_then_line() -> None:
    chunks = (
        _chunk("b.py", 10, "def same(): pass"),
        _chunk("a.py", 20, "def same(): pass"),
        _chunk("a.py", 5, "def same(): pass"),
    )

    results = search_chunks_bm25("same", chunks)

    result_chunks = []
    result_ranks = []
    for item in results:
        result_chunks.append(item.chunk)
        result_ranks.append(item.rank)
    assert result_chunks == [chunks[2], chunks[1], chunks[0]]
    assert result_ranks == [1, 2, 3]


def test_no_hit_and_empty_query_return_empty() -> None:
    chunks = (_chunk("a.py", 1, "def unrelated(): pass"),)

    assert search_chunks_bm25("missing", chunks) == ()
    assert search_chunks_bm25("", chunks) == ()


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_raises(limit: int) -> None:
    with pytest.raises(ValueError, match="正整数"):
        search_chunks_bm25("query", (), limit=limit)
