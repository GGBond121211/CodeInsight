"""Validate the Project-0008 task-6 eval assets and fixture layout."""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
BACKEND_ROOT = EVAL_DIR.parents[1]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "cases.json"

EXPECTED_SCHEMA_VERSION = 1
EXPECTED_FIXTURE = "tests/fixtures/sample_repo"
EXPECTED_CASE_COUNT = 27

REQUIRED_CATEGORIES = {
    "alias",
    "code_document_conflict",
    "cross_file_call",
    "data_flow",
    "insufficient_evidence",
    "partial_evidence",
    "path_clue",
    "same_name_symbol",
    "semantic_paraphrase",
    "symbol_lookup",
}

OLD_CASE_IDS = {
    "symbol-checkout",
    "symbol-create-order",
    "symbol-order-total",
    "symbol-default-currency",
    "call-api-checkout",
    "call-checkout-validation",
    "call-checkout-reserve",
    "call-checkout-price",
    "call-total-unit",
    "flow-payload-request",
    "flow-request-receipt",
    "flow-price-total",
    "insufficient-payment",
}

HARD_CASE_IDS = {
    "hard-duplicate-customer-find-record",
    "hard-duplicate-admin-find-record",
    "hard-docs-collision-find-record",
    "hard-docs-collision-finalize-shipment",
    "hard-call-fulfillment-entry-pipeline",
    "hard-flow-fulfillment-three-path",
    "hard-semantic-paraphrase-dispatch-lane",
    "hard-semantic-paraphrase-quarantine",
    "hard-alias-eta",
    "hard-alias-sla",
    "hard-insufficient-refund-provider",
    "hard-insufficient-refund-delivery",
}

NEW_CASE_IDS = {
    "path-clue-pricing-total",
    "partial-order-request",
}


def _load_cases() -> dict:
    with CASES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_document_shape() -> None:
    document = _load_cases()
    assert document["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert document["fixture"] == EXPECTED_FIXTURE
    cases = document["cases"]
    assert len(cases) == EXPECTED_CASE_COUNT
    ids = []
    for case in cases:
        ids.append(case["id"])
    assert len(ids) == len(set(ids))
    assert OLD_CASE_IDS <= set(ids)
    assert HARD_CASE_IDS <= set(ids)
    assert NEW_CASE_IDS <= set(ids)


def test_required_categories_and_protects() -> None:
    cases = _load_cases()["cases"]
    categories = set()
    for case in cases:
        categories.add(case["category"])
    assert REQUIRED_CATEGORIES <= categories
    for case in cases:
        assert case["id"]
        assert case["category"]
        assert isinstance(case["protects"], str)
        assert case["protects"].strip()
        assert "question" in case["input"] or "path" in case["input"]
        assert case["expected"]["outcome"]


def _normalized_line_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(text.splitlines())


def test_evidence_validity() -> None:
    cases = _load_cases()["cases"]
    for case in cases:
        for item in case["expected"].get("evidence", []):
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


def test_evidence_cases_are_answered() -> None:
    cases = _load_cases()["cases"]
    for case in cases:
        if "evidence" in case["expected"]:
            assert case["expected"]["outcome"] == "answered"


def test_insufficient_evidence_cases_have_no_evidence() -> None:
    cases = _load_cases()["cases"]
    for case in cases:
        if case["category"] == "insufficient_evidence":
            assert case["expected"]["outcome"] == "insufficient_evidence"
            assert "evidence" not in case["expected"]
