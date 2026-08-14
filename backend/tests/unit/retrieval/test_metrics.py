"""检索评测指标单元测试。"""

from dataclasses import FrozenInstanceError

import pytest

from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.metrics import RetrievalMetrics, evaluate_cases


def _chunk(path: str, start_line: int, end_line: int, text: str) -> SourceChunk:
    return SourceChunk(path, start_line, end_line, text)


def _case(case_id: str, question: str, *evidence: dict) -> dict:
    expected: dict = {"outcome": "answered"}
    if evidence:
        expected["evidence"] = list(evidence)
    return {
        "id": case_id,
        "category": "symbol_lookup",
        "input": {"question": question},
        "expected": expected,
    }


def _insufficient_case(case_id: str, question: str) -> dict:
    return {
        "id": case_id,
        "category": "insufficient_evidence",
        "input": {"question": question},
        "expected": {"outcome": "insufficient_evidence"},
    }


def test_single_evidence_covered_at_rank_one() -> None:
    chunk = _chunk(
        "src/shop/pricing.py",
        11,
        14,
        "def order_total(sku: str, quantity: int) -> int:\n"
        "    price = unit_price(sku)\n"
        "    return price * quantity",
    )
    case = _case(
        "single",
        "Where is order_total defined?",
        {"path": "src/shop/pricing.py", "start_line": 11, "end_line": 14},
    )
    metrics = evaluate_cases([case], (chunk,))
    assert metrics.case_count == 1
    assert metrics.applicable_case_count == 1
    assert metrics.top1 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.valid_evidence_rate == 1.0


def test_multi_evidence_requires_all_pieces_across_results() -> None:
    api = _chunk(
        "src/shop/api.py",
        3,
        12,
        "from .models import OrderRequest\nfrom .service import checkout\n"
        "receipt = checkout(request)",
    )
    service = _chunk("src/shop/service.py", 11, 14, "def checkout(request):\n    return Receipt()")
    case = _case(
        "multi",
        "How does the API call checkout?",
        {"path": "src/shop/api.py", "start_line": 3, "end_line": 12},
        {"path": "src/shop/service.py", "start_line": 11, "end_line": 14},
    )
    metrics = evaluate_cases([case], (api, service))
    assert metrics.applicable_case_count == 1
    assert metrics.top1 == 0.0
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.recall_at_5 == 1.0


def test_insufficient_evidence_cases_count_but_do_not_apply() -> None:
    chunk = _chunk("src/shop/config.py", 3, 3, "DEFAULT_CURRENCY = 'USD'")
    cases = [
        _case(
            "answered",
            "Where is DEFAULT_CURRENCY defined?",
            {"path": "src/shop/config.py", "start_line": 3, "end_line": 3},
        ),
        _insufficient_case("insufficient", "Which payment processor is used?"),
    ]
    metrics = evaluate_cases(cases, (chunk,))
    assert metrics.case_count == 2
    assert metrics.applicable_case_count == 1
    assert metrics.top1 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.valid_evidence_rate == 1.0


def test_evidence_at_rank_two_yields_mrr_half() -> None:
    decoy = _chunk("a.py", 1, 1, "checkout")
    target = _chunk("src/shop/service.py", 11, 19, "def checkout(request):")
    case = _case(
        "rank-two",
        "checkout",
        {"path": "src/shop/service.py", "start_line": 11, "end_line": 19},
    )
    metrics = evaluate_cases([case], (decoy, target))
    assert metrics.top1 == 0.0
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.recall_at_5 == 1.0
    assert metrics.valid_evidence_rate == 1.0


def test_missing_paths_and_invalid_lines_lower_evidence_rate() -> None:
    empty_path = _chunk("", 1, 1, "checkout")
    zero_start = _chunk("src/shop/service.py", 0, 5, "checkout")
    valid = _chunk("src/shop/service.py", 11, 19, "def checkout(request):")
    case = _case(
        "validity",
        "checkout",
        {"path": "src/shop/service.py", "start_line": 11, "end_line": 19},
    )
    metrics = evaluate_cases([case], (empty_path, zero_start, valid))
    assert metrics.top1 == 0.0
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.recall_at_5 == 1.0
    assert metrics.valid_evidence_rate == 1 / 3


def test_no_results_gives_zero_metrics() -> None:
    chunk = _chunk("src/shop/config.py", 3, 3, "DEFAULT_CURRENCY = 'USD'")
    case = _case(
        "miss",
        "zzz missing term",
        {"path": "src/shop/config.py", "start_line": 3, "end_line": 3},
    )
    metrics = evaluate_cases([case], (chunk,))
    assert metrics.top1 == 0.0
    assert metrics.mean_reciprocal_rank == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.valid_evidence_rate == 0.0


def test_no_applicable_cases_yields_zero_metrics() -> None:
    cases = [_insufficient_case("insufficient", "Which payment processor is used?")]
    metrics = evaluate_cases(cases, ())
    assert metrics.case_count == 1
    assert metrics.applicable_case_count == 0
    assert metrics.top1 == 0.0
    assert metrics.mean_reciprocal_rank == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.valid_evidence_rate == 0.0


def test_single_chunk_covers_multiple_evidence_in_one_file() -> None:
    chunk = _chunk(
        "src/shop/pricing.py",
        6,
        14,
        "def unit_price(sku: str) -> int:\n"
        "    return PRICE_CENTS[sku]\n"
        "\n"
        "\n"
        "def order_total(sku: str, quantity: int) -> int:\n"
        "    price = unit_price(sku)\n"
        "    return price * quantity",
    )
    case = _case(
        "same-file",
        "unit_price order_total",
        {"path": "src/shop/pricing.py", "start_line": 6, "end_line": 8},
        {"path": "src/shop/pricing.py", "start_line": 11, "end_line": 14},
    )
    metrics = evaluate_cases([case], (chunk,))
    assert metrics.top1 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_5 == 1.0


def test_metrics_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        RetrievalMetrics(1, 1, 1.0, 1.0, 1.0, 1.0).case_count = 2
