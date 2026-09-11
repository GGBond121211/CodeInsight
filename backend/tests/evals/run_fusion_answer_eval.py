"""Run live answer evaluations over the 131-case fusion asset."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from codeinsight.application.answer_repository import answer_repository
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.query_router import route_question
from codeinsight.domain.answer import AutoAnswer, RepositoryAnswer
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.evaluation.answer_metrics import citation_covers, citation_is_valid
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel

# 2026-09-11：旧 LangGraph 路线的导入已冻结；保留原名以便回退：
# from codeinsight.agent.workflow import run_citation_agent
# from codeinsight.domain.agent import AgentRepositoryAnswer

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "multilingual_cases_fusion.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"


def evidence_items(case: dict) -> tuple[dict, ...]:
    expected = case.get("expected", {})
    candidates = list(expected.get("evidence", ()))
    if not candidates:
        for subquestion in expected.get("subquestions", ()):
            subquestion_evidence = subquestion.get("evidence", ())
            for item in subquestion_evidence:
                candidates.append(item)
    unique: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for item in candidates:
        key = (item["path"], item["start_line"], item["end_line"])
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return tuple(unique)


def result_metrics(case: dict, result: RepositoryAnswer | AutoAnswer | None) -> dict:
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
    valid: list[bool] = []
    for citation in result.citations:
        is_valid = citation_is_valid(citation, FIXTURE_ROOT)
        valid.append(is_valid)

    supported: list[bool] = []
    for citation in result.citations:
        citation_is_supported = False
        for item in evidence:
            if citation_covers(citation, item):
                citation_is_supported = True
                break
        supported.append(citation_is_supported)

    evidence_hit: list[bool] = []
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
    return {
        "outcome_hit": result.outcome == expected["outcome"],
        "valid_citation_rate": valid_count / len(valid) if valid else 1.0,
        "citation_precision": supported_count / len(supported) if supported else 1.0,
        "evidence_recall": evidence_hit_count / len(evidence_hit) if evidence_hit else 1.0,
        "failure": ",".join(failures) if failures else None,
    }


def expected_router_labels(case: dict) -> dict:
    category = case["category"]
    return {
        "language": "mixed" if case["language"] == "zh-en" else case["language"],
        "retrieval_mode": "hybrid",
        "execution_route": "agent" if category == "multi_intent" else "linear",
        "subquestion_count": len(case.get("expected", {}).get("subquestions", ())) or 1,
    }


def router_metrics(case: dict, router_result) -> dict:
    expected = expected_router_labels(case)
    plan = router_result.plan
    actual_modes: set[str] = set()
    for item in plan.subquestions:
        actual_modes.add(item.retrieval_mode)
    return {
        "valid_plan": not router_result.used_fallback,
        "language_hit": plan.language == expected["language"],
        "subquestion_count_hit": len(plan.subquestions) == expected["subquestion_count"],
        "retrieval_mode_hit": expected["retrieval_mode"] in actual_modes,
        "execution_route_hit": plan.execution_route == expected["execution_route"],
    }


def _citation_payload(result: RepositoryAnswer | AutoAnswer | None) -> list[dict]:
    if result is None:
        return []
    payload: list[dict] = []
    for citation in result.citations:
        payload.append(
            {
                "evidence_id": citation.evidence_id,
                "relative_path": citation.relative_path,
                "start_line": citation.start_line,
                "end_line": citation.end_line,
            }
        )
    return payload


def _group_names(case: dict) -> tuple[str, ...]:
    groups = [
        "all",
        f"source:{case['source_set']}",
        f"language:{case['language']}",
        f"category:{case['category']}",
    ]
    if case["language"] == "zh-en":
        groups.append("mixed_language")
    if case["category"] == "multi_intent":
        groups.append("multi_question")
    if case["category"] == "semantic_paraphrase":
        groups.append("semantic_mismatch")
    if case["category"] == "noisy_query":
        groups.append("typo_or_noise")
    return tuple(groups)


def run_case(case: dict, mode: str, model, embedding_model) -> dict:
    started = perf_counter()
    usage = Counter()
    router_result = None
    original_router_plan = None
    result: RepositoryAnswer | AutoAnswer | None = None
    # 2026-09-11：旧 LangGraph 路线的结果对象已冻结，这里恒为 None。
    # agent_result: AgentRepositoryAnswer | None = None
    agent_result = None
    error: str | None = None

    def tracked_complete(system_prompt: str, user_prompt: str):
        usage["model_calls"] += 1
        return model.complete(system_prompt, user_prompt)

    def tracked_generate(system_prompt: str, user_prompt: str):
        usage["model_calls"] += 1
        return model.generate(system_prompt, user_prompt)

    def tracked_embed(texts) -> EmbeddingBatch:
        usage["embedding_calls"] += 1
        batch = embedding_model.embed(texts)
        usage["embedding_input_tokens"] += batch.input_tokens or 0
        return batch

    try:
        if mode == "linear":
            result = answer_repository(
                FIXTURE_ROOT,
                case["input"]["question"],
                generate=tracked_generate,
                retrieval_mode="hybrid",
                semantic_embed=tracked_embed,
            )
        elif mode == "router-linear":
            router_result = route_question(case["input"]["question"], complete=tracked_complete)
            original_router_plan = router_result.plan
            plan = replace(
                router_result.plan,
                execution_route="linear",
            )
            router_result = replace(router_result, plan=plan)
            semantic_embed = tracked_embed if embedding_model else None
            result = auto_answer_repository(
                FIXTURE_ROOT,
                router_result=router_result,
                generate=tracked_generate,
                semantic_embed=semantic_embed,
            )
        # 2026-09-11：router-agent 模式已冻结（旧 LangGraph 路线），原实现保留：
        # else:
        #     agent_result = run_citation_agent(
        #         FIXTURE_ROOT,
        #         case["input"]["question"],
        #         complete=tracked_complete,
        #         retrieval_mode="auto",
        #         semantic_embed=semantic_embed,
        #         query_plan=router_result.plan,
        #     )
        #     result = agent_result.result
    except Exception as caught:  # noqa: BLE001 - evaluation records per-case boundaries
        error = f"{type(caught).__name__}: {caught}"

    elapsed = (perf_counter() - started) * 1000
    metrics = result_metrics(case, result)
    router_payload = None
    if router_result is not None:
        router_payload = {
            "used_fallback": router_result.used_fallback,
            "fallback_reason": router_result.fallback_reason,
            "input_tokens": router_result.input_tokens,
            "output_tokens": router_result.output_tokens,
            "plan": original_router_plan.to_dict(),
            "execution_plan": router_result.plan.to_dict(),
            "forced_execution_route": router_result.plan.execution_route,
            **router_metrics(case, replace(router_result, plan=original_router_plan)),
        }
        usage["router_input_tokens"] += router_result.input_tokens or 0
        usage["router_output_tokens"] += router_result.output_tokens or 0

    answer_input_tokens = (getattr(result, "input_tokens", 0) or 0) if result else 0
    answer_output_tokens = (getattr(result, "output_tokens", 0) or 0) if result else 0
    usage["answer_input_tokens"] += answer_input_tokens
    usage["answer_output_tokens"] += answer_output_tokens
    # The repository/Agent result already contains the input-token total returned by
    # the embedding wrapper. Do not add it again here; the wrapper only owns the call
    # count while the domain result owns the provider usage total.
    if isinstance(result, AutoAnswer):
        usage["embedding_input_tokens"] = result.embedding_input_tokens or 0
    if agent_result is not None:
        usage["embedding_input_tokens"] = agent_result.embedding_input_tokens or 0

    agent_steps: list[str] = []
    if agent_result is not None:
        for event in agent_result.events:
            agent_steps.append(event.step)

    return {
        "case_id": case["id"],
        "source_set": case["source_set"],
        "category": case["category"],
        "language": case["language"],
        "mode": mode,
        "question": case["input"]["question"],
        "expected_outcome": case["expected"]["outcome"],
        "outcome": result.outcome if result else None,
        "answer": result.answer if result else None,
        "citations": _citation_payload(result),
        "router": router_payload,
        "model_calls": usage["model_calls"],
        "embedding_calls": usage["embedding_calls"],
        "router_input_tokens": usage["router_input_tokens"],
        "router_output_tokens": usage["router_output_tokens"],
        "answer_input_tokens": answer_input_tokens,
        "answer_output_tokens": answer_output_tokens,
        "embedding_input_tokens": usage["embedding_input_tokens"],
        "elapsed_milliseconds": elapsed,
        "revisions": agent_result.revisions if agent_result else 0,
        "agent_steps": agent_steps,
        "error": error,
        "groups": _group_names(case),
        **metrics,
    }


def summarize(results: list[dict]) -> dict:
    count = len(results)
    router_results: list[dict] = []
    outcome_hits = 0
    valid_citation_total = 0.0
    citation_precision_total = 0.0
    evidence_recall_total = 0.0
    total_model_calls = 0
    total_embedding_calls = 0
    total_router_input_tokens = 0
    total_router_output_tokens = 0
    total_answer_input_tokens = 0
    total_answer_output_tokens = 0
    total_embedding_input_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    elapsed_total = 0.0
    error_count = 0

    for item in results:
        if item["router"]:
            router_results.append(item["router"])
        if item["outcome_hit"]:
            outcome_hits += 1
        valid_citation_total += item["valid_citation_rate"]
        citation_precision_total += item["citation_precision"]
        evidence_recall_total += item["evidence_recall"]
        total_model_calls += item["model_calls"]
        total_embedding_calls += item["embedding_calls"]
        total_router_input_tokens += item["router_input_tokens"]
        total_router_output_tokens += item["router_output_tokens"]
        total_answer_input_tokens += item["answer_input_tokens"]
        total_answer_output_tokens += item["answer_output_tokens"]
        total_embedding_input_tokens += item["embedding_input_tokens"]
        total_input_tokens += item["router_input_tokens"] + item["answer_input_tokens"]
        total_output_tokens += item["router_output_tokens"] + item["answer_output_tokens"]
        elapsed_total += item["elapsed_milliseconds"]
        if item["error"] is not None:
            error_count += 1

    router_case_count = len(router_results)
    valid_plan_count = 0
    language_hit_count = 0
    subquestion_count_hit_count = 0
    retrieval_mode_hit_count = 0
    execution_route_hit_count = 0
    for item in router_results:
        if item["valid_plan"]:
            valid_plan_count += 1
        if item["language_hit"]:
            language_hit_count += 1
        if item["subquestion_count_hit"]:
            subquestion_count_hit_count += 1
        if item["retrieval_mode_hit"]:
            retrieval_mode_hit_count += 1
        if item["execution_route_hit"]:
            execution_route_hit_count += 1

    return {
        "case_count": count,
        "outcome_accuracy": outcome_hits / count if count else 0.0,
        "valid_citation_rate": valid_citation_total / count if count else 0.0,
        "citation_precision": citation_precision_total / count if count else 0.0,
        "evidence_recall": evidence_recall_total / count if count else 0.0,
        "total_model_calls": total_model_calls,
        "total_embedding_calls": total_embedding_calls,
        "total_router_input_tokens": total_router_input_tokens,
        "total_router_output_tokens": total_router_output_tokens,
        "total_answer_input_tokens": total_answer_input_tokens,
        "total_answer_output_tokens": total_answer_output_tokens,
        "total_embedding_input_tokens": total_embedding_input_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_elapsed_milliseconds": elapsed_total / count if count else 0.0,
        "errors": error_count,
        "router": {
            "case_count": router_case_count,
            "valid_plan_rate": valid_plan_count / router_case_count
            if router_case_count
            else None,
            "language_accuracy": language_hit_count / router_case_count
            if router_case_count
            else None,
            "subquestion_split_accuracy": subquestion_count_hit_count / router_case_count
            if router_case_count
            else None,
            "retrieval_mode_accuracy": retrieval_mode_hit_count / router_case_count
            if router_case_count
            else None,
            "execution_route_accuracy": execution_route_hit_count / router_case_count
            if router_case_count
            else None,
        },
    }


def grouped_summary(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        for group in item["groups"]:
            grouped.setdefault(group, []).append(item)
    summaries: dict[str, dict] = {}
    for group, items in sorted(grouped.items()):
        summaries[group] = summarize(items)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("linear", "router-linear", "router-agent"), required=True
    )
    parser.add_argument("--source-set", choices=("legacy31", "task11_v2_100"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--output-suffix")
    args = parser.parse_args()
    if args.mode == "router-agent":
        # 2026-09-11：旧 LangGraph 路线冻结，router-agent 不再运行。
        parser.error("router-agent 模式已冻结：旧 LangGraph 代码理解路线不再运行")
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = document["cases"]
    if args.source_set:
        selected_cases = []
        for case in cases:
            if case["source_set"] == args.source_set:
                selected_cases.append(case)
        cases = selected_cases
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")
    end = None if args.count is None else args.start + args.count
    cases = cases[args.start : end]
    model = OpenAIChatModel.from_environment()
    embedding_model = (
        OpenAIEmbeddingModel.from_environment()
        if os.environ.get("CODEINSIGHT_EMBEDDING_MODEL")
        else None
    )
    results: list[dict] = []
    for case in cases:
        case_result = run_case(case, args.mode, model, embedding_model)
        results.append(case_result)
    selection = {
        "source_set": args.source_set,
        "start": args.start,
        "count": len(cases),
        "requested_count": args.count,
    }
    payload = {
        "schema_version": 1,
        "case_set": document["case_set"],
        "mode": args.mode,
        "selection": selection,
        "model": model.model,
        "embedding_model_configured": embedding_model is not None,
        "base_url_host": urlsplit(os.environ.get("CODEINSIGHT_BASE_URL", "")).hostname,
        "summary": summarize(results),
        "groups": grouped_summary(results),
        "sources": document["sources"],
        "notes": [
            "All 131 cases use the same sample repository and deterministic citation metrics.",
            "Linear uses the fixed hybrid route; Router modes use the model-produced "
            "retrieval plan.",
            "The evaluator records actual model and embedding calls through wrappers.",
            "No LLM judge is used; outcome and citation metrics are deterministic.",
        ],
        "cases": results,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    output_path = OUTPUT_ROOT / f"multilingual-fusion-answer-comparison-{args.mode}{suffix}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output_path), "summary": payload["summary"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
