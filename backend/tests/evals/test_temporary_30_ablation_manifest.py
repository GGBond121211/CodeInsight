"""Contract tests for the frozen Temporary-30 ablation manifest."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from tests.evals.build_temporary_30_ablation_manifest import CONFIRMATION_IDS

EVAL_DIR = Path(__file__).resolve().parent
ASSET_PATH = EVAL_DIR / "multilingual_cases_temporary_30.json"
MANIFEST_PATH = EVAL_DIR / "temporary_30_ablation_manifest.json"
FIXTURE_ROOT = EVAL_DIR.parent / "fixtures" / "sample_repo"


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(ASSET_PATH.read_text(encoding="utf-8")),
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


def test_manifest_freezes_all_cases_subquestions_and_split() -> None:
    asset, manifest = _documents()
    assert manifest["case_count"] == 30
    assert manifest["subquestion_count"] == 38
    manifest_ids = set()
    for case in manifest["cases"]:
        manifest_ids.add(case["case_id"])
    asset_ids = set()
    for case in asset["cases"]:
        asset_ids.add(case["id"])
    assert manifest_ids == asset_ids
    split_values = []
    for case in manifest["cases"]:
        split_values.append(case["split"])
    assert Counter(split_values) == {
        "diagnostic": 20,
        "confirmation": 10,
    }
    confirmation = []
    for case in manifest["cases"]:
        if case["split"] == "confirmation":
            confirmation.append(case)
    confirmation_ids = set()
    confirmation_languages = []
    for case in confirmation:
        confirmation_ids.add(case["case_id"])
        confirmation_languages.append(case["language"])
    assert confirmation_ids == CONFIRMATION_IDS
    assert Counter(confirmation_languages) == {"zh": 5, "zh-en": 5}
    categories = set()
    for case in manifest["cases"]:
        categories.add(case["category"])
    assert len(categories) == 9


def test_scenarios_do_not_cross_splits() -> None:
    _, manifest = _documents()
    splits_by_scenario: defaultdict[str, set[str]] = defaultdict(set)
    for case in manifest["cases"]:
        splits_by_scenario[case["scenario"]].add(case["split"])
    for splits in splits_by_scenario.values():
        assert len(splits) == 1


def test_evidence_claims_and_subquestion_ids_are_valid() -> None:
    _, manifest = _documents()
    claim_ids = []
    for case in manifest["cases"]:
        actual_ids = []
        for item in case["subquestions"]:
            actual_ids.append(item["id"])
        expected_ids = []
        for index in range(1, len(case["subquestions"]) + 1):
            expected_ids.append(f"Q{index}")
        assert actual_ids == expected_ids
        for subquestion in case["subquestions"]:
            if subquestion["expected_outcome"] == "insufficient_evidence":
                assert subquestion["expected_evidence"] == []
            else:
                assert subquestion["expected_evidence"]
            for evidence in subquestion["expected_evidence"]:
                path = FIXTURE_ROOT / evidence["path"]
                assert path.is_file()
                lines = path.read_text(encoding="utf-8").splitlines()
                assert 1 <= evidence["start_line"] <= evidence["end_line"] <= len(lines)
            for claim in subquestion["atomic_claims"]:
                claim_ids.append(claim["claim_id"])
                assert claim["polarity"] in {"true", "false", "unsupported"}
                assert claim["expected_statement"].strip()
    assert len(claim_ids) == len(set(claim_ids)) == 38
