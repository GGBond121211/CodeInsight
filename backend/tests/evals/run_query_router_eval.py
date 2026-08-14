"""Run an explicit real-model Query Router evaluation on Task 11 cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from codeinsight.application.query_router import route_question
from codeinsight.infrastructure.openai_chat import OpenAIChatModel
from codeinsight.prompts.query_router import PROMPT_VERSION

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "multilingual_cases.json"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"


def expected_labels(case: dict) -> dict:
    """Return transparent offline labels used only for Router diagnostics."""
    category = case["category"]
    expected_subquestion_count = len(case.get("expected", {}).get("subquestions", ())) or 1
    return {
        "language": "mixed" if case["language"] == "zh-en" else case["language"],
        "retrieval_mode": "bm25",
        "execution_route": "agent" if category == "multi_intent" else "linear",
        "subquestion_count": expected_subquestion_count,
    }


def _case_payload(case: dict, result) -> dict:
    expected = expected_labels(case)
    plan = result.plan
    actual_modes = {item.retrieval_mode for item in plan.subquestions}
    return {
        "case_id": case["id"],
        "language": case["language"],
        "category": case["category"],
        "used_fallback": result.used_fallback,
        "fallback_reason": result.fallback_reason,
        "plan": plan.to_dict(),
        "expected_labels": expected,
        "language_hit": plan.language == expected["language"],
        "subquestion_count_hit": len(plan.subquestions) == expected["subquestion_count"],
        "retrieval_mode_hit": expected["retrieval_mode"] in actual_modes,
        "execution_route_hit": plan.execution_route == expected["execution_route"],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "elapsed_milliseconds": result.elapsed_milliseconds,
    }


def summarize(case_payloads: list[dict]) -> dict:
    count = len(case_payloads)
    return {
        "case_count": count,
        "valid_plan_rate": sum(not item["used_fallback"] for item in case_payloads) / count
        if count
        else 0.0,
        "language_accuracy": sum(item["language_hit"] for item in case_payloads) / count
        if count
        else 0.0,
        "subquestion_split_accuracy": sum(item["subquestion_count_hit"] for item in case_payloads)
        / count
        if count
        else 0.0,
        "retrieval_mode_accuracy": sum(item["retrieval_mode_hit"] for item in case_payloads) / count
        if count
        else 0.0,
        "execution_route_accuracy": sum(item["execution_route_hit"] for item in case_payloads)
        / count
        if count
        else 0.0,
        "fallback_count": sum(item["used_fallback"] for item in case_payloads),
        "total_input_tokens": sum(item["input_tokens"] for item in case_payloads),
        "total_output_tokens": sum(item["output_tokens"] for item in case_payloads),
        "mean_elapsed_milliseconds": sum(item["elapsed_milliseconds"] for item in case_payloads)
        / count
        if count
        else 0.0,
        "fallback_reasons": dict(
            Counter(item["fallback_reason"] for item in case_payloads if item["fallback_reason"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = [case for case in document["cases"] if not args.case_ids or case["id"] in args.case_ids]
    model = OpenAIChatModel.from_environment()
    results = [
        _case_payload(case, route_question(case["input"]["question"], complete=model.complete))
        for case in cases
    ]
    payload = {
        "schema_version": 1,
        "case_set": document["case_set"],
        "model": model.model,
        "prompt_version": PROMPT_VERSION,
        "model_calls": len(results),
        "summary": summarize(results),
        "cases": results,
        "notes": [
            "This command explicitly calls the configured chat model once per case and "
            "never retries.",
            "Expected labels are deterministic diagnostic labels, not runtime prompt content.",
            "Router output never supplies evidence paths or line numbers.",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_name = (
        "query-router-comparison.json" if not args.case_ids else "query-router-subset.json"
    )
    output_path = OUTPUT_ROOT / output_name
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
