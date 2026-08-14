"""Run the live grounded-answer evaluation and write its reproducible artifact."""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from codeinsight.application.answer_repository import answer_repository
from codeinsight.domain.answer import RepositoryAnswer
from codeinsight.domain.errors import ModelCallError, ModelResponseError
from codeinsight.evaluation.answer_metrics import (
    answer_failure_types,
    evaluate_answer_results,
)
from codeinsight.infrastructure.openai_chat import OpenAIChatModel
from codeinsight.prompts.code_answer import PROMPT_VERSION

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "cases.json"
ANSWER_CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "answer_cases.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"


def load_answer_cases() -> list[dict]:
    """Resolve the focused answer set against the frozen retrieval cases."""
    source_document = json.loads(SOURCE_CASES_PATH.read_text(encoding="utf-8"))
    answer_document = json.loads(ANSWER_CASES_PATH.read_text(encoding="utf-8"))
    source_by_id = {case["id"]: case for case in source_document["cases"]}
    resolved = []
    for selected in answer_document["cases"]:
        case = dict(source_by_id[selected["id"]])
        case["required_terms"] = list(selected["required_terms"])
        resolved.append(case)
    return resolved


def build_evaluation_payload(
    *,
    cases: list[dict],
    results: dict[str, RepositoryAnswer | None],
    errors: dict[str, str],
    elapsed_milliseconds: dict[str, int],
    fixture_root: Path,
    model: str,
    base_url_host: str | None,
    prompt_version: str = PROMPT_VERSION,
) -> dict:
    """Build the stable artifact payload without making a model call."""
    metrics = evaluate_answer_results(cases, results, fixture_root)
    case_payloads = []
    input_tokens = 0
    output_tokens = 0
    for case in cases:
        case_id = case["id"]
        result = results.get(case_id)
        if result:
            input_tokens += result.input_tokens or 0
            output_tokens += result.output_tokens or 0
        case_payloads.append(
            {
                "case_id": case_id,
                "question": case["input"]["question"],
                "expected_outcome": case["expected"]["outcome"],
                "outcome": result.outcome if result else None,
                "answer": result.answer if result else None,
                "citations": [asdict(citation) for citation in result.citations] if result else [],
                "model": result.model if result else model,
                "prompt_version": result.prompt_version if result else prompt_version,
                "input_tokens": result.input_tokens if result else None,
                "output_tokens": result.output_tokens if result else None,
                "elapsed_milliseconds": elapsed_milliseconds[case_id],
                "failure_types": answer_failure_types(case, result, fixture_root),
                "error": errors.get(case_id),
            }
        )

    payload = {
        "prompt_version": prompt_version,
        "retrieval_mode": "bm25",
        "model": model,
        "case_count": len(cases),
        "metrics": asdict(metrics),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "cases": case_payloads,
    }
    if base_url_host:
        payload["base_url_host"] = base_url_host
    return payload


def main() -> int:
    """Run all focused answer cases against the configured live model."""
    cases = load_answer_cases()
    model = OpenAIChatModel.from_environment()
    results: dict[str, RepositoryAnswer | None] = {}
    errors: dict[str, str] = {}
    elapsed_milliseconds: dict[str, int] = {}

    for case in cases:
        started = time.perf_counter()
        try:
            results[case["id"]] = answer_repository(
                FIXTURE_ROOT,
                case["input"]["question"],
                generate=model.generate,
                retrieval_mode="bm25",
                limit=5,
            )
        except (ModelCallError, ModelResponseError, OSError, ValueError) as error:
            results[case["id"]] = None
            errors[case["id"]] = str(error)
        elapsed_milliseconds[case["id"]] = round((time.perf_counter() - started) * 1000)

    configured_url = os.environ.get("CODEINSIGHT_BASE_URL", "")
    base_url_host = urlsplit(configured_url).hostname if configured_url else None
    payload = build_evaluation_payload(
        cases=cases,
        results=results,
        errors=errors,
        elapsed_milliseconds=elapsed_milliseconds,
        fixture_root=FIXTURE_ROOT,
        model=model.model,
        base_url_host=base_url_host,
        prompt_version=PROMPT_VERSION,
    )
    output_name = "answer-baseline.json" if PROMPT_VERSION == "code-answer-v1" else "answer-v2.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / output_name).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "prompt_version": payload["prompt_version"],
        "retrieval_mode": payload["retrieval_mode"],
        "model": payload["model"],
        "case_count": payload["case_count"],
        "metrics": payload["metrics"],
        "usage": payload["usage"],
        "failed_case_ids": [item["case_id"] for item in payload["cases"] if item["failure_types"]],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
