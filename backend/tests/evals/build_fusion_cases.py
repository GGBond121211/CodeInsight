"""Restore the legacy Task 11 asset when needed and build the 131-case fusion set."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
LEGACY_PATH = EVAL_DIR / "multilingual_cases_v1.json"
CURRENT_PATH = EVAL_DIR / "multilingual_cases.json"
FUSION_PATH = EVAL_DIR / "multilingual_cases_fusion.json"
LEGACY_BLOB = "1f396118a0f0082f59e41ca1332d4a9d337e46cd"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _restore_legacy_asset() -> dict:
    if LEGACY_PATH.exists():
        return _load_json(LEGACY_PATH)
    payload = subprocess.check_output(
        ["git", "cat-file", "-p", LEGACY_BLOB],
        cwd=EVAL_DIR.parents[2],
    )
    LEGACY_PATH.write_bytes(payload)
    return json.loads(payload)


def _tag_cases(cases: list[dict], source_set: str) -> list[dict]:
    tagged: list[dict] = []
    for case in cases:
        item = deepcopy(case)
        item["source_set"] = source_set
        item["source_case_id"] = case["id"]
        tagged.append(item)
    return tagged


def build_fusion_document() -> dict:
    legacy = _restore_legacy_asset()
    current = _load_json(CURRENT_PATH)
    legacy_cases = _tag_cases(legacy["cases"], "legacy31")
    current_cases = _tag_cases(current["cases"], "task11_v2_100")
    cases = legacy_cases + current_cases

    ids = []
    questions = []
    for case in cases:
        ids.append(case["id"])
        questions.append(case["input"]["question"])
    if len(cases) != 131:
        raise ValueError(f"expected 131 cases, got {len(cases)}")
    if len(ids) != len(set(ids)):
        raise ValueError("fusion case IDs are not unique")
    if len(questions) != len(set(questions)):
        raise ValueError("fusion questions are not unique")

    return {
        "schema_version": 1,
        "case_set": "task11_multilingual_fusion_v1",
        "fixture": current["fixture"],
        "description": (
            "Fusion evaluation asset containing the recovered Task 11 v1 legacy set "
            "and the Task 11 v2 high-difficulty 100-case set."
        ),
        "sources": [
            {
                "source_set": "legacy31",
                "case_set": legacy["case_set"],
                "case_count": len(legacy_cases),
                "asset": "tests/evals/multilingual_cases_v1.json",
            },
            {
                "source_set": "task11_v2_100",
                "case_set": current["case_set"],
                "case_count": len(current_cases),
                "asset": "tests/evals/multilingual_cases.json",
            },
        ],
        "generation_contract": {
            "case_count": 131,
            "source_case_counts": {"legacy31": 31, "task11_v2_100": 100},
            "deduplication": "unique id and unique input question across both source assets",
            "evidence_contract": "preserve each source case expected outcome and evidence exactly",
        },
        "cases": cases,
    }


def main() -> int:
    document = build_fusion_document()
    FUSION_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(FUSION_PATH),
                "case_count": len(document["cases"]),
                "sources": document["sources"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
