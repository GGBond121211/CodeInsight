"""Integration coverage for opt-in hybrid retrieval."""

from pathlib import Path

from codeinsight.application.search_repository import search_repository
from codeinsight.domain.semantic import EmbeddingBatch

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


def test_hybrid_can_recover_a_semantic_business_phrase() -> None:
    results = search_repository(
        FIXTURE_ROOT,
        "易碎件要走专门的配送通道",
        limit=5,
        retrieval_mode="hybrid",
        semantic_embed=_fake_multilingual_embedder,
    )

    assert results
    assert any(item.chunk.relative_path == "src/shop/shipping/workflow.py" for item in results)
    assert any(item.retrieval_reason in {"semantic_match", "hybrid_match"} for item in results)


def test_hybrid_requires_an_explicit_embedding_function() -> None:
    try:
        search_repository(
            FIXTURE_ROOT,
            "Where is checkout defined?",
            retrieval_mode="hybrid",
        )
    except ValueError as error:
        assert "embedding model" in str(error)
    else:
        raise AssertionError("hybrid retrieval should require an embedding model")
