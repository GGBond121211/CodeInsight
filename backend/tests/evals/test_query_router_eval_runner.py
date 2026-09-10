"""Offline contract tests for Query Router evaluation labels and metrics."""

from tests.evals.run_query_router_eval import expected_labels, summarize


def test_expected_labels_keep_hybrid_and_multi_intent_gates_explicit() -> None:
    semantic = {
        "language": "zh",
        "category": "semantic_paraphrase",
        "expected": {},
    }
    multi = {
        "language": "zh-en",
        "category": "multi_intent",
        "expected": {"subquestions": [{}, {}, {}]},
    }

    assert expected_labels(semantic) == {
        "language": "zh",
        "retrieval_mode": "hybrid",
        "execution_route": "linear",
        "subquestion_count": 1,
    }
    assert expected_labels(multi)["language"] == "mixed"
    assert expected_labels(multi)["execution_route"] == "agent"
    assert expected_labels(multi)["subquestion_count"] == 3


def test_router_summary_is_zero_safe_and_counts_fallbacks() -> None:
    payload = {
        "used_fallback": True,
        "fallback_reason": "router_invalid_or_low_confidence",
        "language_hit": False,
        "subquestion_count_hit": True,
        "retrieval_mode_hit": True,
        "execution_route_hit": True,
        "input_tokens": 4,
        "output_tokens": 2,
        "elapsed_milliseconds": 10.0,
    }
    summary = summarize([payload])

    assert summary["fallback_count"] == 1
    assert summary["total_input_tokens"] == 4
    assert summary["fallback_reasons"] == {"router_invalid_or_low_confidence": 1}
    assert summarize([])["case_count"] == 0
