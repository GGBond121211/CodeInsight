"""Frozen asset checks for the HTTPX grounded-answer comparison."""

import json
from pathlib import Path

CASES_PATH = Path(__file__).with_name("httpx_answer_cases.json")
SOURCE_PATH = Path(__file__).with_name("httpx_cases.json")
FOCUS_IDS = {
    "httpx-flow-top-level-get",
    "httpx-flow-sync-transport",
    "httpx-flow-async-transport",
}


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(CASES_PATH.read_text(encoding="utf-8")),
        json.loads(SOURCE_PATH.read_text(encoding="utf-8")),
    )


def test_httpx_answer_cases_are_frozen_before_live_run() -> None:
    document, source_document = _documents()
    source_ids = {case["id"] for case in source_document["cases"]}
    resolved_ids = [case.get("source_case_id", case.get("id")) for case in document["cases"]]

    assert document["schema_version"] == 1
    assert document["source_cases"] == "tests/evals/httpx_cases.json"
    assert len(resolved_ids) == 8
    assert len(set(resolved_ids)) == 8
    assert FOCUS_IDS <= set(resolved_ids)
    assert all(
        case.get("source_case_id") in source_ids
        for case in document["cases"]
        if "source_case_id" in case
    )


def test_httpx_answer_cases_cover_required_business_shapes() -> None:
    document, source_document = _documents()
    source_by_id = {case["id"]: case for case in source_document["cases"]}
    resolved = [
        source_by_id[case["source_case_id"]] if "source_case_id" in case else case
        for case in document["cases"]
    ]
    categories = {case["category"] for case in resolved}
    insufficient = [case for case in document["cases"] if "source_case_id" not in case]

    assert {
        "cross_file_call",
        "same_name_symbol",
        "symbol_lookup",
        "insufficient_evidence",
    } <= categories
    assert any(case["id"] == "httpx-redirect-decision" for case in resolved)
    assert len(insufficient) == 1
    assert insufficient[0]["expected"] == {
        "outcome": "insufficient_evidence",
        "evidence": [],
    }
    for selected in document["cases"]:
        if selected.get("source_case_id"):
            assert selected["required_terms"]
        else:
            assert selected["required_terms"] == []
