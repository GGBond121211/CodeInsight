"""数据集准入门的回归测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_quality import (  # noqa: E402
    load_mapping,
    validate_conversation_document,
    validate_policy,
)

POLICY_PATH = ROOT / "experiments" / "configs" / "dataset_quality_policy_2_1_0.yaml"
BASELINE_PATH = ROOT / "experiments" / "configs" / "baseline_2_1_0.yaml"
DATASET_PATH = ROOT / "backend" / "tests" / "evals" / "conversation_cases_2_1_0.json"
FIXTURE_ROOT = ROOT / "backend" / "tests" / "fixtures" / "sample_repo"


def _document() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_policy_accepts_all_registered_assets() -> None:
    report = validate_policy(POLICY_PATH, baseline_path=BASELINE_PATH, repo_root=ROOT)

    assert report.accepted
    conversation = next(item for item in report.checks if item.dataset_id == "conversation_2_1_0")
    assert conversation.state == "READY_FOR_MODEL_PILOT"
    assert conversation.actual_count == 24
    assert conversation.actual_turn_count == 123
    assert conversation.metrics["semantic_review_coverage"] == 1.0
    assert conversation.metrics["coverage_ready"] is True


def test_invalid_evidence_range_blocks_dataset() -> None:
    document = _document()
    document["cases"][0]["turns"][0]["expected"]["evidence"][0]["end_line"] = 10_000

    check = validate_conversation_document(
        document, FIXTURE_ROOT, expected_count=24, expected_turn_count=123
    )

    assert not check.accepted
    assert any(issue.code == "evidence_out_of_range" for issue in check.issues)


def test_duplicate_case_id_blocks_dataset() -> None:
    document = _document()
    document["cases"][1]["id"] = document["cases"][0]["id"]

    check = validate_conversation_document(
        document, FIXTURE_ROOT, expected_count=24, expected_turn_count=123
    )

    assert not check.accepted
    assert any(issue.code == "duplicate_id" for issue in check.issues)


def test_holdout_is_not_tuning_input() -> None:
    manifest = load_mapping(ROOT / "experiments" / "configs" / "experiment_splits_2_1_0.yaml")
    policy = manifest["split_policy"]

    assert "holdout" not in policy["tuning"]
    assert policy["final_only"] == ["holdout"]


@pytest.mark.parametrize(
    "mutation,expected_code",
    [
        (
            lambda document: document["cases"][0]["turns"][0]["expected"].pop("must_retain"),
            "context_contract",
        ),
        (
            lambda document: document["cases"][0]["turns"][0]["expected"].update(
                {"raw_response": "forbidden"}
            ),
            "raw_model_content",
        ),
    ],
)
def test_context_contract_and_raw_model_content_are_gated(mutation, expected_code: str) -> None:
    document = _document()
    mutation(document)

    check = validate_conversation_document(
        document, FIXTURE_ROOT, expected_count=24, expected_turn_count=123
    )

    assert any(issue.code == expected_code for issue in check.issues)
