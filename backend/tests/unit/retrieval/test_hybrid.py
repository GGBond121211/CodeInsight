"""稳定稀疏/语义融合测试。"""

import pytest

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.hybrid import fuse_ranked_chunks, rerank_ranked_chunks


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
        limit=2,
    )

    assert results[0].chunk.relative_path == "src/service.py"
    assert results[0].retrieval_reason == "hybrid_match"
