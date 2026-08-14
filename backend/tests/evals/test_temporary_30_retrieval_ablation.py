"""Focused tests for Temporary-30 retrieval ablation mechanics."""

from __future__ import annotations

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from tests.evals.run_temporary_30_retrieval_ablation import _metrics


def _ranked(path: str, start: int, end: int, rank: int = 1) -> RankedChunk:
    return RankedChunk(
        chunk=SourceChunk(
            relative_path=path,
            start_line=start,
            end_line=end,
            text="evidence",
        ),
        score=1.0,
        rank=rank,
    )


def test_metrics_distinguish_source_fusion_and_rerank_loss() -> None:
    expected = [{"path": "a.py", "start_line": 3, "end_line": 4}]
    correct = _ranked("a.py", 1, 8)
    wrong = _ranked("b.py", 1, 8)
    metrics = _metrics(expected, (correct, wrong), (wrong,), (wrong,))
    assert metrics["raw_source_candidate_recall"] == 1.0
    assert metrics["fused_candidate_recall_at_20"] == 0.0
    assert metrics["fusion_retention"] is False
    assert metrics["conditional_rerank_success"] is None

    metrics = _metrics(expected, (correct,), (correct,), (wrong,))
    assert metrics["fused_candidate_recall_at_20"] == 1.0
    assert metrics["evidence_recall_at_5"] == 0.0
    assert metrics["conditional_rerank_success"] is False


def test_insufficient_subquestion_is_not_given_fake_recall() -> None:
    metrics = _metrics([], (), (), ())
    assert metrics["raw_source_candidate_recall"] is None
    assert metrics["evidence_recall_at_5"] is None
