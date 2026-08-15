"""Tests for structured Temporary-30 answer and statistical contracts."""

from __future__ import annotations

import json

import pytest

from codeinsight.domain.errors import ModelResponseError
from tests.evals.summarize_temporary_30_ablation import (
    _blind_review_pack,
    summarize_answer_rows,
)
from tests.evals.temporary_30_ablation import (
    StructuredCaseAnswer,
    StructuredCaseReview,
    StructuredSubquestionAnswer,
    StructuredSubquestionReview,
    case_cluster_bootstrap,
    changed_subquestion_ids,
    exact_mcnemar,
    majority,
    parse_structured_answer,
    parse_structured_review,
)


def test_structured_answer_requires_exact_ids_and_local_evidence() -> None:
    payload = {
        "case_id": "case-1",
        "subquestion_outputs": [
            {"subquestion_id": "Q1", "outcome": "answered", "answer": "a", "citations": ["Q1E1"]},
            {
                "subquestion_id": "Q2",
                "outcome": "insufficient_evidence",
                "answer": "b",
                "citations": [],
            },
        ],
    }
    answer = parse_structured_answer(
        json.dumps(payload),
        case_id="case-1",
        expected_subquestion_ids=("Q1", "Q2"),
        allowed_evidence={"Q1": frozenset({"Q1E1"}), "Q2": frozenset({"Q2E1"})},
        model="fake",
        input_tokens=1,
        output_tokens=2,
    )
    subquestion_ids = []
    for item in answer.subquestions:
        subquestion_ids.append(item.subquestion_id)
    assert subquestion_ids == ["Q1", "Q2"]
    payload["subquestion_outputs"][0]["citations"] = ["Q2E1"]
    with pytest.raises(ModelResponseError, match="another group"):
        parse_structured_answer(
            json.dumps(payload),
            case_id="case-1",
            expected_subquestion_ids=("Q1", "Q2"),
            allowed_evidence={"Q1": frozenset({"Q1E1"}), "Q2": frozenset({"Q2E1"})},
            model="fake",
            input_tokens=1,
            output_tokens=2,
        )


def test_review_allows_empty_supported_citations_for_insufficient() -> None:
    review = parse_structured_review(
        json.dumps(
            {
                "subquestion_reviews": [
                    {
                        "subquestion_id": "Q1",
                        "verdict": "pass",
                        "supported_citations": [],
                        "feedback": "Repository cannot prove it.",
                    }
                ]
            }
        ),
        expected_subquestion_ids=("Q1",),
        allowed_evidence={"Q1": frozenset()},
        model="fake",
        input_tokens=1,
        output_tokens=1,
    )
    assert review.reviews[0].verdict == "pass"


def test_unauthorized_revision_is_detected() -> None:
    before = StructuredCaseAnswer(
        "c",
        (
            StructuredSubquestionAnswer("Q1", "answered", "before", ("Q1E1",)),
            StructuredSubquestionAnswer("Q2", "answered", "same", ("Q2E1",)),
        ),
        "fake",
        0,
        0,
        "",
    )
    after = StructuredCaseAnswer(
        "c",
        (
            StructuredSubquestionAnswer("Q1", "answered", "changed", ("Q1E1",)),
            StructuredSubquestionAnswer("Q2", "answered", "also changed", ("Q2E1",)),
        ),
        "fake",
        0,
        0,
        "",
    )
    review = StructuredCaseReview(
        (
            StructuredSubquestionReview("Q1", "revise", ("Q1E1",), "fix"),
            StructuredSubquestionReview("Q2", "pass", ("Q2E1",), "keep"),
        ),
        "fake",
        0,
        0,
        "",
    )
    assert changed_subquestion_ids(before, after, review) == ("Q2",)


def test_case_cluster_bootstrap_averages_all_repetitions_inside_case() -> None:
    result = case_cluster_bootstrap(
        {"a": (1.0, 1.0, 1.0), "b": (-1.0, -1.0, -1.0)}, iterations=100, seed=7
    )
    assert result["case_count"] == 2
    assert result["mean"] == 0.0


def test_case_level_binary_statistics() -> None:
    assert majority((True, False, True)) is True
    result = exact_mcnemar((False, True, False), (True, True, False))
    assert result == {"wins": 1, "losses": 0, "ties": 2, "p_value": 1.0}


def test_summary_preserves_case_repeats_as_one_cluster() -> None:
    rows = []
    for case_id, outcomes in {"a": (False, True, True), "b": (False, False, False)}.items():
        for repeat, outcome in enumerate(outcomes, start=1):
            for group in ("B00", "B01", "B10", "B11"):
                value = outcome if group in {"B01", "B11"} else False
                rows.append(
                    {
                        "case_id": case_id,
                        "split": "confirmation",
                        "language": "zh",
                        "category": "test",
                        "repeat": repeat,
                        "group": group,
                        "error": None,
                        "metrics": {
                            "automated_grounded_pass": value,
                            "human_verified_complete_pass": None,
                            "outcome_accuracy": float(value),
                            "citation_precision": float(value),
                            "evidence_recall": float(value),
                        },
                        "elapsed_milliseconds": 1.0,
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                    }
                )
    summary = summarize_answer_rows(rows)
    effect = summary["effects"]["critic_without_semantic"]
    assert effect["cluster_bootstrap"]["case_count"] == 2
    assert effect["case_level_binary"]["wins"] == 1


def test_blind_pack_hides_identity_but_keeps_private_mapping() -> None:
    manifest = {
        "cases": [
            {
                "case_id": "case-1",
                "original_question": "question",
                "subquestions": [{"id": "Q1", "question": "sub", "atomic_claims": []}],
            }
        ]
    }
    row = {
        "case_id": "case-1",
        "split": "confirmation",
        "repeat": 1,
        "group": "B10",
        "error": None,
        "draft": {"case_id": "case-1", "subquestion_outputs": []},
        "final": {"case_id": "case-1", "subquestion_outputs": []},
        "evidence_groups": [],
    }
    pack, mapping = _blind_review_pack([row], manifest)
    assert "case_id" not in pack["entries"][0]
    assert "group" not in pack["entries"][0]
    assert mapping["entries"] == [
        {
            "review_id": "R0001",
            "case_id": "case-1",
            "repeat": 1,
            "group": "B10",
            "stage": "draft",
        }
    ]
