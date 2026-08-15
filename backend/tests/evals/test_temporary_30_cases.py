"""Contract checks for the temporary 30-case dense evaluation asset."""

import json
from collections import Counter
from pathlib import Path

from tests.evals.build_temporary_30_cases import SELECTED_CASE_IDS

ASSET_PATH = Path(__file__).with_name("multilingual_cases_temporary_30.json")


def _load() -> dict:
    return json.loads(ASSET_PATH.read_text(encoding="utf-8"))


def test_temporary_asset_has_stable_high_difficulty_coverage() -> None:
    document = _load()
    cases = document["cases"]
    assert document["case_set"] == "task11_multilingual_temporary_30"
    case_ids = []
    for case in cases:
        case_ids.append(case["id"])
    assert case_ids == list(SELECTED_CASE_IDS)
    assert len(cases) == 30
    languages = []
    categories = []
    for case in cases:
        languages.append(case["language"])
        categories.append(case["category"])
    assert Counter(languages) == {"zh": 15, "zh-en": 15}
    assert Counter(categories) == {
        "ambiguity": 3,
        "boundary_behavior": 3,
        "code_document_conflict": 3,
        "cross_file_trace": 4,
        "data_flow": 4,
        "insufficient_evidence": 2,
        "multi_intent": 4,
        "noisy_query": 3,
        "semantic_paraphrase": 4,
    }


def test_temporary_asset_preserves_expected_contracts() -> None:
    for case in _load()["cases"]:
        expected = case["expected"]
        evidence = expected.get("evidence", ())
        if "subquestions" in expected:
            evidence_items = []
            for subquestion in expected["subquestions"]:
                for item in subquestion.get("evidence", ()):
                    evidence_items.append(item)
            evidence = tuple(evidence_items)
        if expected["outcome"] == "insufficient_evidence":
            assert not evidence
        else:
            assert evidence
        assert case["difficulty"] == "very_hard"
        assert len(case["challenge_features"]) >= 5
