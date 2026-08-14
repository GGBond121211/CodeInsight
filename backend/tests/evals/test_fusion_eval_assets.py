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
    assert Counter(case["source_set"] for case in cases) == {
        "legacy31": 31,
        "task11_v2_100": 100,
    }
    assert len({case["id"] for case in cases}) == 131
    assert len({case["input"]["question"] for case in cases}) == 131


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
    assert Counter(case["language"] for case in cases) == {
        "zh-en": 65,
        "zh": 59,
        "en": 7,
    }
    assert sum(bool(_evidence(case)) for case in cases) == 120
    assert sum(not _evidence(case) for case in cases) == 11
