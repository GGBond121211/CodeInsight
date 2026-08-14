"""Auto Answer 使用的共享检索引擎集成测试。"""

from pathlib import Path

import pytest

from codeinsight.application.search_repository import search_repository
from codeinsight.domain.semantic import EmbeddingBatch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


def _fake_embed(texts):
    return EmbeddingBatch("fake", tuple((1.0, 0.0) for _ in texts), len(texts))


@pytest.mark.parametrize("retrieval_mode", ["lexical", "bm25"])
def test_internal_sparse_retrieval_resolves_within_fixture(retrieval_mode: str) -> None:
    results = search_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        limit=5,
        retrieval_mode=retrieval_mode,
    )

    assert "src/shop/service.py" in [item.chunk.relative_path for item in results]


def test_internal_hybrid_combines_bm25_and_semantic_candidates() -> None:
    results = search_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        limit=5,
        retrieval_mode="hybrid",
        semantic_embed=_fake_embed,
    )

    assert results
    expected_reasons = {"direct_match", "semantic_match", "hybrid_match"}
    assert all(item.retrieval_reason in expected_reasons for item in results)


@pytest.mark.parametrize("removed_mode", ["ast-bm25", "graph-bm25"])
def test_removed_retrieval_modes_are_rejected(removed_mode: str) -> None:
    with pytest.raises(ValueError, match="不支持的检索模式"):
        search_repository(FIXTURE_ROOT, "Where is checkout?", retrieval_mode=removed_mode)
