"""确定性词法排序测试。"""

import pytest

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.lexical import score_chunk, search_chunks
from codeinsight.retrieval.tokenizer import tokenize


def _chunk(path: str, start_line: int, text: str) -> SourceChunk:
    return SourceChunk(path, start_line, start_line, text)


def test_direct_identifier_and_path_hits() -> None:
    target = _chunk("src/shop/pricing.py", 5, "def apply_discount(amount):")
    other = _chunk("src/shop/models.py", 2, "class PricingConfig:")
    results = search_chunks("apply_discount", (target, other))
    assert isinstance(results[0], RankedChunk)
    assert results[0].chunk is target
    assert score_chunk(tokenize("pricing"), target) > 0.0


def test_filename_bonus_and_code_body_beat_document() -> None:
    code = _chunk("src/shop/config.py", 1, "def apply_discount(amount): return amount")
    doc = _chunk("docs/shipping-guide.md", 1, "def apply_discount(amount): return amount")
    directory = _chunk("config/util.py", 1, "x = 1")
    assert score_chunk(tokenize("apply_discount"), code) > score_chunk(
        tokenize("apply_discount"), doc
    )
    assert score_chunk(tokenize("config"), code) > score_chunk(tokenize("config"), directory)


def test_ties_sort_by_path_then_line_and_ranks_are_continuous() -> None:
    chunks = (
        _chunk("b.py", 10, "def same(): pass"),
        _chunk("a.py", 20, "def same(): pass"),
        _chunk("a.py", 5, "def same(): pass"),
    )
    results = search_chunks("same", chunks)
    result_chunks = []
    result_ranks = []
    for item in results:
        result_chunks.append(item.chunk)
        result_ranks.append(item.rank)
    assert result_chunks == [chunks[2], chunks[1], chunks[0]]
    assert result_ranks == [1, 2, 3]


def test_limit_zero_scores_and_empty_query() -> None:
    chunks = (_chunk("a.py", 1, "def unrelated(): pass"), _chunk("b.py", 1, "shared"))
    assert search_chunks("missing", chunks) == ()
    assert search_chunks("shared", chunks, limit=1)[0].chunk is chunks[1]
    assert search_chunks("", chunks) == ()


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_raises(limit: int) -> None:
    with pytest.raises(ValueError, match="正整数"):
        search_chunks("query", (), limit=limit)
