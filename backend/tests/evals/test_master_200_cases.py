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
    assert len({case["id"] for case in cases}) == 200
    assert Counter(case["repository_id"] for case in cases) == {
        "sample_repo": 100,
        "httpx": 30,
        "click": 35,
        "requests": 35,
    }

    removable = {"repository_id", "source_set"}
    restored_old = [
        {key: value for key, value in case.items() if key not in removable} for case in cases[:100]
    ]
    assert restored_old == source["cases"]

    new_cases = cases[100:]
    assert all(case["language"] in {"zh", "zh-en"} for case in new_cases)
    assert all(case["difficulty"] == "very_hard" for case in new_cases)
    assert all(len(case["challenge_features"]) >= 5 for case in new_cases)
    assert all(case["expected"]["outcome"] == "answered" for case in new_cases)
    assert all(case["expected"]["required_terms"] for case in new_cases)


def test_repository_fingerprints_and_every_evidence_span() -> None:
    master = load(MASTER)
    repositories = {item["id"]: item for item in master["repositories"]}
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
    questions = [case["input"]["question"] for case in cases]
    assert len(set(questions)) == 100
    assert all(any("\u4e00" <= char <= "\u9fff" for char in question) for question in questions)
    assert all(len(question) >= 45 for question in questions)


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
            assert all(segment["path"] == requirement["path"] for segment in segments)
            assert all(
                segment["end_line"] - segment["start_line"] + 1 <= 80 for segment in segments
            )
            assert all(
                left["end_line"] + 1 == right["start_line"]
                for left, right in zip(segments, segments[1:])
            )
            flattened.extend(segments)
        assert flattened == expected["evidence"]
