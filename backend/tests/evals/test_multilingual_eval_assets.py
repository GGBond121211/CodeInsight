"""Validate the Task 11.1 multilingual and noisy evaluation asset."""

import json
import re
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).parent
BACKEND_ROOT = EVAL_DIR.parents[1]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
CASES_PATH = EVAL_DIR / "multilingual_cases.json"
BASE_CASES_PATH = EVAL_DIR / "cases.json"

EXPECTED_SCHEMA_VERSION = 2
EXPECTED_FIXTURE = "tests/fixtures/sample_repo"
EXPECTED_CASE_COUNT = 100
EXPECTED_SCENARIO_COUNT = 25
REQUIRED_LANGUAGES = {"zh", "zh-en"}
REQUIRED_CATEGORIES = {
    "ambiguity",
    "boundary_behavior",
    "code_document_conflict",
    "cross_file_trace",
    "data_flow",
    "insufficient_evidence",
    "multi_intent",
    "noisy_query",
    "semantic_paraphrase",
}
ALLOWED_OUTCOMES = {"answered", "insufficient_evidence", "partially_answered"}
EXPECTED_CATEGORY_COUNTS = {
    "ambiguity": 12,
    "boundary_behavior": 12,
    "code_document_conflict": 8,
    "cross_file_trace": 12,
    "data_flow": 12,
    "insufficient_evidence": 8,
    "multi_intent": 12,
    "noisy_query": 12,
    "semantic_paraphrase": 12,
}
REQUIRED_CHALLENGE_FEATURES = {
    "contradictory_claim",
    "false_premise",
    "irrelevant_context",
    "log_fragment",
    "mixed_intent",
    "omitted_subject",
    "scope_confusion",
    "self_correction",
    "temporal_confusion",
    "typo_noise",
    "uncertain_causality",
}
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _load_cases() -> dict:
    with CASES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _iter_evidence(expected: dict):
    yield from expected.get("evidence", [])
    for subquestion in expected.get("subquestions", []):
        yield from subquestion.get("evidence", [])


def _normalized_line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_document_shape_and_coverage() -> None:
    document = _load_cases()
    assert document["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert document["fixture"] == EXPECTED_FIXTURE
    assert document["case_set"] == "task11_multilingual_realistic_v2"
    assert len(document["research_basis"]) >= 6
    assert document["generation_contract"]["case_count"] == EXPECTED_CASE_COUNT
    assert document["generation_contract"]["scenario_count"] == EXPECTED_SCENARIO_COUNT
    assert document["generation_contract"]["forbidden_languages"] == ["en"]

    cases = document["cases"]
    assert len(cases) == EXPECTED_CASE_COUNT
    ids = []
    for case in cases:
        ids.append(case["id"])
    assert len(ids) == len(set(ids))
    for case_id in ids:
        assert case_id.startswith("t11-v2-")

    languages = set()
    language_values = []
    categories = []
    scenarios_list = []
    for case in cases:
        languages.add(case["language"])
        language_values.append(case["language"])
        categories.append(case["category"])
        scenarios_list.append(case["scenario"])
    assert languages == REQUIRED_LANGUAGES
    assert language_values.count("zh") == 50
    assert language_values.count("zh-en") == 50
    assert Counter(categories) == EXPECTED_CATEGORY_COUNTS
    assert set(EXPECTED_CATEGORY_COUNTS) == REQUIRED_CATEGORIES

    scenarios = Counter(scenarios_list)
    assert len(scenarios) == EXPECTED_SCENARIO_COUNT
    assert set(scenarios.values()) == {4}
    for scenario in scenarios:
        variants = []
        for case in cases:
            if case["scenario"] == scenario:
                variants.append(case)
        variant_languages = []
        for case in variants:
            variant_languages.append(case["language"])
        assert Counter(variant_languages) == {"zh": 2, "zh-en": 2}


def test_cases_are_nontrivial_and_do_not_duplicate_frozen_cases() -> None:
    document = _load_cases()
    with BASE_CASES_PATH.open(encoding="utf-8") as handle:
        frozen_cases = json.load(handle)["cases"]
    frozen_questions = set()
    for case in frozen_cases:
        frozen_questions.add(case["input"].get("question"))
    questions = []
    for case in document["cases"]:
        questions.append(case["input"]["question"])
    assert len(questions) == len(set(questions))

    for case in document["cases"]:
        question = case["input"]["question"]
        assert len(question.strip()) >= 50
        assert CHINESE_RE.search(question)
        if case["language"] == "zh-en":
            assert LATIN_RE.search(question)
        assert question not in frozen_questions
        assert case["difficulty"] == "very_hard"
        assert case["protects"].strip()
        assert case["design_notes"]
        assert len(case["challenge_features"]) >= 5


def test_cases_cover_realistic_conflict_and_noise_patterns() -> None:
    cases = _load_cases()["cases"]
    features = []
    for case in cases:
        for feature in set(case["challenge_features"]):
            features.append(feature)
    feature_counts = Counter(features)

    assert REQUIRED_CHALLENGE_FEATURES <= set(feature_counts)
    assert feature_counts["mixed_language"] == 50
    assert feature_counts["typo_noise"] >= 30
    assert feature_counts["false_premise"] >= 25
    assert feature_counts["scope_confusion"] >= 25
    assert feature_counts["irrelevant_context"] >= 25
    assert feature_counts["self_correction"] >= 25
    assert feature_counts["omitted_subject"] >= 25
    assert feature_counts["temporal_confusion"] >= 25


def test_evidence_paths_and_lines_are_valid() -> None:
    for case in _load_cases()["cases"]:
        expected = case["expected"]
        assert expected["outcome"] in ALLOWED_OUTCOMES
        for item in _iter_evidence(expected):
            raw_path = item["path"]
            assert isinstance(raw_path, str)
            assert "\\" not in raw_path
            relative = Path(raw_path)
            assert not relative.is_absolute()
            assert ".." not in relative.parts
            target = FIXTURE_ROOT / relative
            assert target.is_file()
            start = item["start_line"]
            end = item["end_line"]
            assert isinstance(start, int) and start >= 1
            assert isinstance(end, int)
            assert start <= end
            assert end <= _normalized_line_count(target)


def test_outcome_contracts_are_consistent() -> None:
    for case in _load_cases()["cases"]:
        expected = case["expected"]
        outcome = expected["outcome"]
        subquestions = expected.get("subquestions", [])

        if outcome == "answered":
            assert expected.get("evidence") or subquestions
            if subquestions:
                for item in subquestions:
                    assert item["outcome"] == "answered"

        if outcome == "insufficient_evidence":
            assert not list(_iter_evidence(expected))
            assert not subquestions

        if outcome == "partially_answered":
            assert subquestions
            sub_outcomes = set()
            for item in subquestions:
                sub_outcomes.add(item["outcome"])
            assert "answered" in sub_outcomes
            assert "insufficient_evidence" in sub_outcomes

        for subquestion in subquestions:
            assert subquestion["question"].strip()
            assert subquestion["outcome"] in ALLOWED_OUTCOMES
