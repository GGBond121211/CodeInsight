"""可选 hybrid 检索集成测试。"""

from pathlib import Path

from codeinsight.application.search_repository import search_repository
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


def _fake_multilingual_embedder(texts):
    vectors = []
    for text in texts:
        if "配送通道" in text or "choose_dispatch_lane" in text:
            vectors.append((1.0, 0.0))
        else:
            vectors.append((0.0, 1.0))
    return EmbeddingBatch("fake-multilingual", tuple(vectors), 4)


class _FakeReranker:
    def rerank(self, query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


def test_hybrid_can_recover_a_semantic_business_phrase() -> None:
    results = search_repository(
        FIXTURE_ROOT,
        "易碎件要走专门的配送通道",
        limit=5,
        retrieval_mode="hybrid",
        semantic_embed=_fake_multilingual_embedder,
        reranker=_FakeReranker(),
    )

    assert results
    has_shipping_workflow = False
    has_semantic_reason = False
    for item in results:
        if item.chunk.relative_path == "src/shop/shipping/workflow.py":
            has_shipping_workflow = True
        if item.retrieval_reason in {"semantic_match", "hybrid_match"}:
            has_semantic_reason = True
    assert has_shipping_workflow
    assert has_semantic_reason


def test_hybrid_requires_an_explicit_embedding_function() -> None:
    try:
        search_repository(
            FIXTURE_ROOT,
            "Where is checkout defined?",
            retrieval_mode="hybrid",
        )
    except ValueError as error:
        assert "Embedding 模型" in str(error)
    else:
        raise AssertionError("hybrid 检索必须需要 Embedding 模型")
