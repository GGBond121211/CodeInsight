"""Contract tests for the recovered 31 + current 100 case fusion asset."""

import json
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).parent
FUSION_PATH = EVAL_DIR / "multilingual_cases_fusion.json"


def _load() -> dict:
    return json.loads(FUSION_PATH.read_text(encoding="utf-8"))


def _evidence(case: dict) -> list[dict]:
    expected = case.get("expected", {})
    items = list(expected.get("evidence", []))
    for subquestion in expected.get("subquestions", []):
        items.extend(subquestion.get("evidence", []))
    return items


def test_fusion_has_both_sources_and_unique_cases() -> None:
    document = _load()
    cases = document["cases"]
    assert document["case_set"] == "task11_multilingual_fusion_v1"
    assert len(cases) == 131
    source_sets = []
    case_ids = set()
    questions = set()
    for case in cases:
        source_sets.append(case["source_set"])
        case_ids.add(case["id"])
        questions.add(case["input"]["question"])
    assert Counter(source_sets) == {
        "legacy31": 31,
        "task11_v2_100": 100,
    }
    assert len(case_ids) == 131
    assert len(questions) == 131


def test_fusion_preserves_source_evidence_contracts() -> None:
    document = _load()
    for case in document["cases"]:
        expected = case["expected"]
        if expected["outcome"] == "insufficient_evidence":
            assert not _evidence(case)
        else:
            assert _evidence(case)


def test_fusion_counts_are_quantifiable() -> None:
    document = _load()
    cases = document["cases"]
    languages = []
    evidence_flags = []
    for case in cases:
        languages.append(case["language"])
        evidence_flags.append(bool(_evidence(case)))
    assert Counter(languages) == {
        "zh-en": 65,
        "zh": 59,
        "en": 7,
    }
    assert sum(evidence_flags) == 120
    missing_evidence_count = 0
    for flag in evidence_flags:
        if not flag:
            missing_evidence_count += 1
    assert missing_evidence_count == 11
