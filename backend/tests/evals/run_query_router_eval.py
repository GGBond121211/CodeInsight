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
        "retrieval_mode": "hybrid",
        "execution_route": "agent" if category == "multi_intent" else "linear",
        "subquestion_count": expected_subquestion_count,
    }


def _case_payload(case: dict, result) -> dict:
    expected = expected_labels(case)
    plan = result.plan
    actual_modes = set()
    for item in plan.subquestions:
        actual_modes.add(item.retrieval_mode)
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
    valid_plan_count = 0
    language_hit_count = 0
    subquestion_count_hit_count = 0
    retrieval_mode_hit_count = 0
    execution_route_hit_count = 0
    fallback_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    elapsed_total = 0
    fallback_reasons = []
    for item in case_payloads:
        if not item["used_fallback"]:
            valid_plan_count += 1
        if item["language_hit"]:
            language_hit_count += 1
        if item["subquestion_count_hit"]:
            subquestion_count_hit_count += 1
        if item["retrieval_mode_hit"]:
            retrieval_mode_hit_count += 1
        if item["execution_route_hit"]:
            execution_route_hit_count += 1
        if item["used_fallback"]:
            fallback_count += 1
        total_input_tokens += item["input_tokens"]
        total_output_tokens += item["output_tokens"]
        elapsed_total += item["elapsed_milliseconds"]
        if item["fallback_reason"]:
            fallback_reasons.append(item["fallback_reason"])

    denominator = count if count else 1
    fallback_reason_counts = Counter(fallback_reasons)
    return {
        "case_count": count,
        "valid_plan_rate": valid_plan_count / denominator if count else 0.0,
        "language_accuracy": language_hit_count / denominator if count else 0.0,
        "subquestion_split_accuracy": subquestion_count_hit_count / denominator if count else 0.0,
        "retrieval_mode_accuracy": retrieval_mode_hit_count / denominator if count else 0.0,
        "execution_route_accuracy": execution_route_hit_count / denominator if count else 0.0,
        "fallback_count": fallback_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_elapsed_milliseconds": elapsed_total / denominator if count else 0.0,
        "fallback_reasons": dict(fallback_reason_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = []
    for case in document["cases"]:
        if not args.case_ids or case["id"] in args.case_ids:
            cases.append(case)
    model = OpenAIChatModel.from_environment()
    results = []
    for case in cases:
        route_result = route_question(case["input"]["question"], complete=model.complete)
        results.append(_case_payload(case, route_result))
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
