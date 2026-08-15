from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tests.evals.build_master_200_cases import PROJECT_ROOT, fingerprint

EVAL_ROOT = Path(__file__).resolve().parent
MASTER = EVAL_ROOT / "master_200_cases.json"
SOURCE = EVAL_ROOT / "multilingual_cases.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_master_200_contract_and_provenance() -> None:
    master = load(MASTER)
    source = load(SOURCE)
    cases = master["cases"]

    assert master["schema_version"] == 1
    assert len(cases) == 200
    case_ids = set()
    repository_ids = []
    for case in cases:
        case_ids.add(case["id"])
        repository_ids.append(case["repository_id"])
    assert len(case_ids) == 200
    assert Counter(repository_ids) == {
        "sample_repo": 100,
        "httpx": 30,
        "click": 35,
        "requests": 35,
    }

    removable = {"repository_id", "source_set"}
    restored_old = []
    for case in cases[:100]:
        restored_case = {}
        for key, value in case.items():
            if key not in removable:
                restored_case[key] = value
        restored_old.append(restored_case)
    assert restored_old == source["cases"]

    new_cases = cases[100:]
    for case in new_cases:
        assert case["language"] in {"zh", "zh-en"}
        assert case["difficulty"] == "very_hard"
        assert len(case["challenge_features"]) >= 5
        assert case["expected"]["outcome"] == "answered"
        assert case["expected"]["required_terms"]


def test_repository_fingerprints_and_every_evidence_span() -> None:
    master = load(MASTER)
    repositories = {}
    for item in master["repositories"]:
        repositories[item["id"]] = item
    assert set(repositories) == {"sample_repo", "httpx", "click", "requests"}

    line_counts: dict[Path, int] = {}
    for metadata in repositories.values():
        root = PROJECT_ROOT / metadata["local_root"]
        assert root.is_dir()
        assert fingerprint(root) == metadata["source_fingerprint"]

    for case in master["cases"]:
        root = PROJECT_ROOT / repositories[case["repository_id"]]["local_root"]
        expected = case["expected"]
        evidence_items = list(expected.get("evidence", []))
        for subquestion in expected.get("subquestions", []):
            evidence_items.extend(subquestion.get("evidence", []))
        if expected["outcome"] == "answered":
            assert evidence_items, case["id"]
        for item in evidence_items:
            path = root / item["path"]
            assert path.is_file(), (case["id"], path)
            if path not in line_counts:
                line_counts[path] = len(path.read_text(encoding="utf-8").splitlines())
            assert 1 <= item["start_line"] <= item["end_line"] <= line_counts[path], (
                case["id"],
                item,
                line_counts[path],
            )


def test_new_questions_are_unique_and_not_clean_english_prompts() -> None:
    cases = load(MASTER)["cases"][100:]
    questions = []
    for case in cases:
        questions.append(case["input"]["question"])
    assert len(set(questions)) == 100
    for question in questions:
        contains_chinese = False
        for char in question:
            if "\u4e00" <= char <= "\u9fff":
                contains_chinese = True
                break
        assert contains_chinese
        assert len(question) >= 45


def test_real_repository_evidence_requirements_are_chunk_compatible() -> None:
    cases = load(MASTER)["cases"][100:]
    for case in cases:
        expected = case["expected"]
        requirements = expected["evidence_requirements"]
        flattened = []
        for requirement in requirements:
            segments = requirement["segments"]
            assert segments
            assert segments[0]["start_line"] == requirement["start_line"]
            assert segments[-1]["end_line"] == requirement["end_line"]
            for segment in segments:
                assert segment["path"] == requirement["path"]
                assert segment["end_line"] - segment["start_line"] + 1 <= 80
            for left, right in zip(segments, segments[1:]):
                assert left["end_line"] + 1 == right["start_line"]
            flattened.extend(segments)
        assert flattened == expected["evidence"]
