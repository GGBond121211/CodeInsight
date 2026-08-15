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
    source_ids = set()
    for case in source_document["cases"]:
        source_ids.add(case["id"])
    resolved_ids = []
    for case in document["cases"]:
        resolved_ids.append(case.get("source_case_id", case.get("id")))

    assert document["schema_version"] == 1
    assert document["source_cases"] == "tests/evals/httpx_cases.json"
    assert len(resolved_ids) == 8
    assert len(set(resolved_ids)) == 8
    assert FOCUS_IDS <= set(resolved_ids)
    for case in document["cases"]:
        if "source_case_id" in case:
            assert case.get("source_case_id") in source_ids


def test_httpx_answer_cases_cover_required_business_shapes() -> None:
    document, source_document = _documents()
    source_by_id = {}
    for case in source_document["cases"]:
        source_by_id[case["id"]] = case
    resolved = []
    insufficient = []
    for case in document["cases"]:
        if "source_case_id" in case:
            resolved.append(source_by_id[case["source_case_id"]])
        else:
            resolved.append(case)
            insufficient.append(case)
    categories = set()
    for case in resolved:
        categories.add(case["category"])

    assert {
        "cross_file_call",
        "same_name_symbol",
        "symbol_lookup",
        "insufficient_evidence",
    } <= categories
    has_redirect_decision = False
    for case in resolved:
        if case["id"] == "httpx-redirect-decision":
            has_redirect_decision = True
            break
    assert has_redirect_decision
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
