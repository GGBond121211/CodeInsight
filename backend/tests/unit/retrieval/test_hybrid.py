"""稳定稀疏/语义融合测试。"""

import pytest

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.infrastructure.reranker import RerankResult
from codeinsight.retrieval.hybrid import fuse_ranked_chunks, rerank_ranked_chunks


class _FakeReranker:
    def rerank(self, query, documents, *, top_n):
        order = sorted(
            range(len(documents)),
            key=lambda index: ("CheckoutService.checkout" not in documents[index], index),
        )
        return tuple(
            RerankResult(index, float(top_n - rank))
            for rank, index in enumerate(order[:top_n])
        )


def _result(path: str, rank: int, reason: str) -> RankedChunk:
    return RankedChunk(
        SourceChunk(path, 1, 3, path),
        score=float(10 - rank),
        rank=rank,
        retrieval_reason=reason,
    )


def test_fusion_rewards_evidence_seen_by_multiple_retrievers() -> None:
    results = fuse_ranked_chunks(
        (
            (
                "bm25",
                (_result("src/a.py", 1, "direct_match"), _result("src/b.py", 2, "direct_match")),
            ),
            ("semantic", (_result("src/b.py", 1, "semantic_match"),)),
        ),
        limit=2,
    )

    result_paths = []
    for item in results:
        result_paths.append(item.chunk.relative_path)
    assert result_paths == ["src/b.py", "src/a.py"]
    assert results[0].retrieval_reason == "hybrid_match"
    assert results[0].rank == 1


def test_fusion_is_stable_and_validates_limit() -> None:
    sources = (("semantic", (_result("src/z.py", 1, "semantic_match"),)),)
    assert fuse_ranked_chunks(sources, limit=1)[0].retrieval_reason == "semantic_match"
    with pytest.raises(ValueError, match="limit 必须是正整数"):
        fuse_ranked_chunks(sources, limit=0)


def test_model_reranker_receives_fused_candidates_and_controls_final_order() -> None:
    class RecordingReranker:
        def __init__(self):
            self.documents = ()

        def rerank(self, query, documents, *, top_n):
            self.documents = tuple(documents)
            return (RerankResult(1, 0.99), RerankResult(0, 0.10))[:top_n]

    reranker = RecordingReranker()
    results = rerank_ranked_chunks(
        "checkout",
        (
            (
                "bm25",
                (
                    _result("src/a.py", 1, "direct_match"),
                    _result("src/b.py", 2, "direct_match"),
                ),
            ),
        ),
        reranker=reranker,
        limit=2,
    )
    assert len(reranker.documents) == 2
    assert "Path: src/a.py" in reranker.documents[0]
    assert [item.chunk.relative_path for item in results] == ["src/b.py", "src/a.py"]


def test_model_reranker_receives_the_post_fusion_candidate_limit() -> None:
    class RecordingReranker:
        def __init__(self):
            self.document_count = 0

        def rerank(self, query, documents, *, top_n):
            self.document_count = len(documents)
            return tuple(RerankResult(index, 1.0) for index in range(top_n))

    reranker = RecordingReranker()
    source = tuple(_result(f"src/{index}.py", index + 1, "direct_match") for index in range(120))
    rerank_ranked_chunks(
        "candidate limit",
        (("bm25", source), ("semantic", source)),
        reranker=reranker,
        limit=10,
        candidate_limit=100,
    )

    assert reranker.document_count == 100


def test_code_aware_reranker_prefers_exact_symbol_identity() -> None:
    symbol = RankedChunk(
        SourceChunk("src/service.py", 10, 15, "def checkout(): pass", "CheckoutService.checkout"),
        score=1.0,
        rank=2,
    )
    generic = RankedChunk(
        SourceChunk("docs/checkout.md", 1, 5, "checkout overview"),
        score=1.0,
        rank=1,
    )

    results = rerank_ranked_chunks(
        "Where is CheckoutService.checkout implemented?",
        (("bm25", (generic, symbol)),),
        reranker=_FakeReranker(),
        limit=2,
    )

    assert results[0].chunk.symbol_path == "CheckoutService.checkout"


def test_code_aware_reranker_rewards_sparse_semantic_agreement() -> None:
    sparse = RankedChunk(
        SourceChunk("src/service.py", 10, 15, "def dispatch(): pass", "dispatch"),
        score=1.0,
        rank=2,
        retrieval_reason="direct_match",
    )
    semantic_copy = RankedChunk(
        sparse.chunk,
        score=0.9,
        rank=1,
        retrieval_reason="semantic_match",
    )
    plain = RankedChunk(
        SourceChunk("src/other.py", 1, 3, "dispatch helper"),
        score=1.0,
        rank=1,
        retrieval_reason="direct_match",
    )

    results = rerank_ranked_chunks(
        "Trace the dispatch call flow",
        (("bm25", (plain, sparse)), ("semantic", (semantic_copy,))),
        reranker=_FakeReranker(),
        limit=2,
    )

    assert results[0].chunk.relative_path == "src/service.py"
    assert results[0].retrieval_reason == "hybrid_match"
