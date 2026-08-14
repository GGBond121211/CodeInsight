import json
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
ANSWER_CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "answer_cases.json"
SOURCE_CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "cases.json"


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(ANSWER_CASES_PATH.read_text(encoding="utf-8")),
        json.loads(SOURCE_CASES_PATH.read_text(encoding="utf-8")),
    )


def test_answer_case_schema_and_ids_are_frozen() -> None:
    answer_document, source_document = _documents()
    cases = answer_document["cases"]
    ids = [case["id"] for case in cases]
    source_ids = {case["id"] for case in source_document["cases"]}

    assert answer_document["schema_version"] == 1
    assert answer_document["source_cases"] == "tests/evals/cases.json"
    assert len(cases) == 13
    assert len(set(ids)) == 13
    assert set(ids) <= source_ids


def test_answer_cases_cover_outcomes_terms_and_business_categories() -> None:
    answer_document, source_document = _documents()
    selected = {case["id"]: case for case in answer_document["cases"]}
    source = {case["id"]: case for case in source_document["cases"]}
    insufficient_ids = {
        case_id
        for case_id, case in source.items()
        if case["expected"]["outcome"] == "insufficient_evidence"
    }

    assert insufficient_ids == {
        "insufficient-payment",
        "hard-insufficient-refund-provider",
        "hard-insufficient-refund-delivery",
    }
    assert insufficient_ids <= selected.keys()
    for case_id, case in selected.items():
        if case_id in insufficient_ids:
            assert case["required_terms"] == []
        else:
            assert case["required_terms"]

    categories = {source[case_id]["category"] for case_id in selected}
    assert {
        "symbol_lookup",
        "cross_file_call",
        "data_flow",
        "same_name_symbol",
        "code_document_conflict",
        "semantic_paraphrase",
        "alias",
        "partial_evidence",
        "insufficient_evidence",
    } <= categories
