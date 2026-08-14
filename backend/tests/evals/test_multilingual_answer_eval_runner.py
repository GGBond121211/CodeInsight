"""Offline contract tests for the Task 11.11 answer runner."""

from types import SimpleNamespace

from tests.evals.run_multilingual_answer_eval import _result_metrics, _summary


def test_answer_summary_treats_missing_model_usage_as_zero() -> None:
    result = SimpleNamespace(outcome="insufficient_evidence", citations=())
    case = {
        "expected": {"outcome": "insufficient_evidence", "evidence": []},
    }
    metrics = _result_metrics(case, result)  # type: ignore[arg-type]
    payload = {
        "answer_input_tokens": None,
        "answer_output_tokens": None,
        "outcome_hit": metrics["outcome_hit"],
        "valid_citation_rate": metrics["valid_citation_rate"],
        "citation_precision": metrics["citation_precision"],
        "evidence_recall": metrics["evidence_recall"],
        "error": None,
    }

    assert _summary([payload])["total_answer_input_tokens"] == 0
    assert _summary([payload])["total_answer_output_tokens"] == 0
