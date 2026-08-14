"""Deterministic contract tests for the Task 11.2 retrieval runner."""

import json
from pathlib import Path

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from tests.evals.run_multilingual_retrieval_eval import (
    RETRIEVAL_MODES,
    aggregate_case_payloads,
    build_comparison,
    case_groups,
    evidence_items,
)

EVAL_DIR = Path(__file__).parent
CASES_PATH = EVAL_DIR / "multilingual_cases.json"
OUTPUT_PATH = EVAL_DIR.parents[2] / "outputs" / "evals" / "multilingual-retrieval-comparison.json"


def _load_document() -> dict:
    with CASES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _fake_search(root, question, *, limit, chunk_max_lines, retrieval_mode):
    del root, chunk_max_lines, retrieval_mode
    chunk = SourceChunk("src/shop/service.py", 10, 19, "def checkout...", "checkout")
    return tuple(
        RankedChunk(chunk=chunk, score=1.0, rank=rank, retrieval_reason="direct_match")
        for rank in range(1, min(limit, 1) + 1)
    )


def test_evidence_items_flatten_subquestion_evidence_without_duplicates() -> None:
    cases = _load_document()["cases"]
    partial = next(case for case in cases if case["id"] == "t11-v2-multi-record-refund-a")
    evidence = evidence_items(partial)
    assert len(evidence) == 2
    assert {item["path"] for item in evidence} == {
        "src/shop/customer/records.py",
        "src/shop/admin/records.py",
    }


def test_case_groups_cover_required_task_11_dimensions() -> None:
    cases = _load_document()["cases"]
    mixed = next(case for case in cases if case["language"] == "zh-en")
    noisy = next(case for case in cases if case["category"] == "noisy_query")
    multi = next(case for case in cases if case["category"] == "multi_intent")
    semantic = next(case for case in cases if case["category"] == "semantic_paraphrase")

    assert "mixed_language" in case_groups(mixed)
    assert "typo_or_noise" in case_groups(noisy)
    assert "multi_question" in case_groups(multi)
    assert "semantic_mismatch" in case_groups(semantic)
    assert "exact_identifier" in case_groups(mixed)


def test_insufficient_cases_are_excluded_from_retrieval_scores() -> None:
    cases = _load_document()["cases"]
    payload = build_comparison(_load_document(), search_fn=_fake_search)
    assert payload["retrieval_modes"] == list(RETRIEVAL_MODES)
    assert payload["applicable_case_count"] == 92
    for mode in RETRIEVAL_MODES:
        metrics = payload["modes"][mode]["strict_metrics"]
        assert metrics["case_count"] == 100
        assert metrics["applicable_case_count"] == 92
        assert metrics["failure_counts"].get("insufficient_evidence", 0) == 0
    assert sum(not evidence_items(case) for case in cases) == 8


def test_aggregate_metrics_are_zero_safe() -> None:
    metrics = aggregate_case_payloads(
        [
            {
                "applicable": False,
                "results": [],
                "retrieval_reason_counts": {},
                "failure_class": "insufficient_evidence",
                "top1_hit": False,
                "coverage_rank_at_5": None,
                "recall_at_5": False,
                "context_lines_at_5": 0,
                "evidence_line_recall_at_5": 0.0,
                "evidence_density_at_5": 0.0,
            }
        ]
    )
    assert metrics["applicable_case_count"] == 0
    assert metrics["top1"] == 0.0
    assert metrics["recall_at_5"] == 0.0


def test_output_is_a_new_task_11_artifact() -> None:
    assert OUTPUT_PATH.name == "multilingual-retrieval-comparison.json"
    assert OUTPUT_PATH.name not in {"retrieval-comparison.json", "ast-retrieval-comparison.json"}
