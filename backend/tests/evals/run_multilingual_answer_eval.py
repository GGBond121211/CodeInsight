"""Run explicit linear, Router-linear, or Router-Agent answer evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.application.answer_repository import answer_repository
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.query_router import route_question
from codeinsight.domain.answer import AutoAnswer, RepositoryAnswer
from codeinsight.evaluation.answer_metrics import citation_covers, citation_is_valid
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "multilingual_cases.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"


def evidence_items(case: dict) -> tuple[dict, ...]:
    expected = case.get("expected", {})
    if expected.get("evidence"):
        return tuple(expected["evidence"])
    unique: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for subquestion in expected.get("subquestions", ()):
        for item in subquestion.get("evidence", ()):
            key = (item["path"], item["start_line"], item["end_line"])
            if key not in seen:
                unique.append(item)
                seen.add(key)
    return tuple(unique)


def _result_metrics(case: dict, result: RepositoryAnswer | AutoAnswer | None) -> dict:
    expected = case["expected"]
    evidence = evidence_items(case)
    if result is None:
        return {
            "outcome_hit": False,
            "valid_citation_rate": 0.0,
            "citation_precision": 0.0,
            "evidence_recall": 0.0,
            "failure": "model_or_embedding_error",
        }
    valid = []
    for citation in result.citations:
        valid.append(citation_is_valid(citation, FIXTURE_ROOT))

    supported = []
    for citation in result.citations:
        citation_is_supported = False
        for item in evidence:
            if citation_covers(citation, item):
                citation_is_supported = True
                break
        supported.append(citation_is_supported)

    evidence_hit = []
    for item in evidence:
        item_is_covered = False
        for citation in result.citations:
            if citation_covers(citation, item):
                item_is_covered = True
                break
        evidence_hit.append(item_is_covered)

    valid_count = 0
    for is_valid in valid:
        if is_valid:
            valid_count += 1
    supported_count = 0
    for is_supported in supported:
        if is_supported:
            supported_count += 1
    evidence_hit_count = 0
    for was_hit in evidence_hit:
        if was_hit:
            evidence_hit_count += 1
    all_valid = True
    for is_valid in valid:
        if not is_valid:
            all_valid = False
            break
    all_evidence_hit = True
    for was_hit in evidence_hit:
        if not was_hit:
            all_evidence_hit = False
            break
    failures: list[str] = []
    if result.outcome != expected["outcome"]:
        failures.append("outcome")
    if not all_valid or not all_evidence_hit:
        failures.append("citation")
    if not failures:
        failure = None
    else:
        failure = ",".join(failures)
    return {
        "outcome_hit": result.outcome == expected["outcome"],
        "valid_citation_rate": valid_count / len(valid) if valid else 1.0,
        "citation_precision": supported_count / len(supported) if supported else 1.0,
        "evidence_recall": evidence_hit_count / len(evidence_hit) if evidence_hit else 1.0,
        "failure": failure,
    }


def _run_case(case: dict, mode: str, model, embedding_model) -> dict:
    started = perf_counter()
    router_result = None
    error = None
    result = None
    try:
        if mode == "linear":
            result = answer_repository(
                FIXTURE_ROOT,
                case["input"]["question"],
                generate=model.generate,
                retrieval_mode="hybrid",
                semantic_embed=embedding_model.embed,
            )
        else:
            router_result = route_question(case["input"]["question"], complete=model.complete)
            plan = replace(
                router_result.plan,
                execution_route="linear" if mode == "router-linear" else "agent",
            )
            router_result = replace(router_result, plan=plan)
            semantic_embed = embedding_model.embed if embedding_model else None
            if mode == "router-linear":
                result = auto_answer_repository(
                    FIXTURE_ROOT,
                    router_result=router_result,
                    generate=model.generate,
                    semantic_embed=semantic_embed,
                )
            else:
                result = run_citation_agent(
                    FIXTURE_ROOT,
                    case["input"]["question"],
                    complete=model.complete,
                    retrieval_mode="auto",
                    semantic_embed=semantic_embed,
                    query_plan=router_result.plan,
                ).result
    except Exception as caught:  # noqa: BLE001 - an eval records per-case boundary failures
        error = type(caught).__name__ + ": " + str(caught)

    metrics = _result_metrics(case, result)
    return {
        "case_id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "mode": mode,
        "router": (
            {
                "used_fallback": router_result.used_fallback,
                "fallback_reason": router_result.fallback_reason,
                "input_tokens": router_result.input_tokens,
                "output_tokens": router_result.output_tokens,
                "elapsed_milliseconds": router_result.elapsed_milliseconds,
                "plan": router_result.plan.to_dict(),
            }
            if router_result
            else None
        ),
        "answer_input_tokens": (getattr(result, "input_tokens", 0) or 0) if result else 0,
        "answer_output_tokens": (getattr(result, "output_tokens", 0) or 0) if result else 0,
        "elapsed_milliseconds": (perf_counter() - started) * 1000,
        "error": error,
        **metrics,
    }


def _summary(results: list[dict]) -> dict:
    count = len(results)
    outcome_hits = 0
    valid_citation_total = 0.0
    citation_precision_total = 0.0
    evidence_recall_total = 0.0
    total_answer_input_tokens = 0
    total_answer_output_tokens = 0
    error_count = 0
    for item in results:
        if item["outcome_hit"]:
            outcome_hits += 1
        valid_citation_total += item["valid_citation_rate"]
        citation_precision_total += item["citation_precision"]
        evidence_recall_total += item["evidence_recall"]
        total_answer_input_tokens += item["answer_input_tokens"] or 0
        total_answer_output_tokens += item["answer_output_tokens"] or 0
        if item["error"] is not None:
            error_count += 1
    return {
        "case_count": count,
        "outcome_accuracy": outcome_hits / count if count else 0.0,
        "valid_citation_rate": valid_citation_total / count if count else 0.0,
        "citation_precision": citation_precision_total / count if count else 0.0,
        "evidence_recall": evidence_recall_total / count if count else 0.0,
        "total_answer_input_tokens": total_answer_input_tokens,
        "total_answer_output_tokens": total_answer_output_tokens,
        "errors": error_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("linear", "router-linear", "router-agent"),
        default="router-linear",
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = []
    for case in document["cases"]:
        if not args.case_ids or case["id"] in args.case_ids:
            cases.append(case)
    model = OpenAIChatModel.from_environment()
    embedding_model = (
        OpenAIEmbeddingModel.from_environment()
        if os.environ.get("CODEINSIGHT_EMBEDDING_MODEL")
        else None
    )
    results = []
    for case in cases:
        results.append(_run_case(case, args.mode, model, embedding_model))
    payload = {
        "schema_version": 1,
        "case_set": document["case_set"],
        "mode": args.mode,
        "model": model.model,
        "summary": _summary(results),
        "cases": results,
        "notes": [
            "The command explicitly calls the configured model and never retries.",
            "No LLM judge is used; outcome and evidence metrics are deterministic.",
            "Embedding is only constructed when CODEINSIGHT_EMBEDDING_MODEL is explicitly set.",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_name = (
        f"multilingual-answer-comparison-{args.mode}.json"
        if not args.case_ids
        else f"multilingual-answer-{args.mode}-subset.json"
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
