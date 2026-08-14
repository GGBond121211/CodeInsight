"""Run the frozen Master-200 through the current public Auto Answer workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.query_router import route_question
from codeinsight.domain.answer import AutoAnswer, RepositoryAnswer, SubQuestionAnswer
from codeinsight.evaluation.answer_metrics import (
    citation_covers,
    citation_is_valid,
    citation_overlaps,
    citations_jointly_cover,
    expected_evidence_requirements,
)
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
CASES_PATH = Path(__file__).with_name("master_200_cases.json")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evals" / "master-200"


@dataclass
class CallUsage:
    chat_calls: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    embedding_calls: int = 0
    embedding_input_tokens: int = 0


class TrackedModels:
    def __init__(self, chat: OpenAIChatModel, embedding: OpenAIEmbeddingModel) -> None:
        self.chat = chat
        self.embedding = embedding
        self.usage = CallUsage()

    def complete(self, system_prompt: str, user_prompt: str):
        result = self.chat.complete(system_prompt, user_prompt)
        self.usage.chat_calls += 1
        self.usage.chat_input_tokens += result.input_tokens or 0
        self.usage.chat_output_tokens += result.output_tokens or 0
        return result

    def generate(self, system_prompt: str, user_prompt: str):
        result = self.chat.generate(system_prompt, user_prompt)
        self.usage.chat_calls += 1
        self.usage.chat_input_tokens += result.input_tokens or 0
        self.usage.chat_output_tokens += result.output_tokens or 0
        return result

    def embed(self, texts):
        result = self.embedding.embed(texts)
        self.usage.embedding_calls += 1
        self.usage.embedding_input_tokens += result.input_tokens or 0
        return result


def _all_expected_evidence(case: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    expected = case["expected"]
    values = list(expected.get("evidence", ()))
    for subquestion in expected.get("subquestions", ()):
        values.extend(subquestion.get("evidence", ()))
    unique = {}
    for item in values:
        unique[(item["path"], item["start_line"], item["end_line"])] = item
    return tuple(unique.values())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _citation_dict(citation) -> dict[str, Any]:
    return {
        "evidence_id": citation.evidence_id,
        "relative_path": citation.relative_path,
        "start_line": citation.start_line,
        "end_line": citation.end_line,
    }


def _answer_payload(result: RepositoryAnswer | AutoAnswer) -> dict[str, Any]:
    return {
        "outcome": result.outcome,
        "answer": result.answer,
        "citations": [_citation_dict(item) for item in result.citations],
        "subquestions": [
            {
                "question": item.question,
                "intent": item.intent,
                "retrieval_mode": item.retrieval_mode,
                "outcome": item.outcome,
                "answer": item.answer,
                "citations": [_citation_dict(citation) for citation in item.citations],
            }
            for item in getattr(result, "subquestions", ())
        ],
    }


def _metrics(case: dict[str, Any], result, root: Path, error: str | None) -> dict[str, Any]:
    expected = case["expected"]
    evidence = _all_expected_evidence(case)
    if result is None:
        return {
            "outcome_hit": False,
            "citation_validity": 0.0,
            "citation_precision": 0.0,
            "evidence_recall": 0.0,
            "complete_evidence_coverage": False,
            "required_terms_pass": False,
            "structured_subquestions": False,
            "subquestion_count_hit": False,
            "automated_grounded_pass": False,
            "failure_types": ["service_or_model_error"],
        }

    citations = result.citations
    requirements = expected_evidence_requirements(expected)
    valid = [citation_is_valid(item, root) for item in citations]
    supported = [
        any(citation_overlaps(item, target) for target in requirements) for item in citations
    ]
    hits = [any(citation_covers(item, target) for item in citations) for target in evidence]
    requirement_hits = [citations_jointly_cover(citations, target) for target in requirements]
    expected_outcome = expected["outcome"]
    outcome_hit = result.outcome == expected_outcome
    if citations:
        citation_validity = sum(valid) / len(valid)
        citation_precision = sum(supported) / len(supported) if evidence else 0.0
    else:
        citation_validity = 1.0
        citation_precision = 1.0 if expected_outcome == "insufficient_evidence" else 0.0
    evidence_recall = sum(hits) / len(hits) if hits else 1.0
    complete_coverage = all(requirement_hits)
    required_terms = expected.get("required_terms", ())
    required_terms_pass = all(
        term.casefold() in result.answer.casefold() for term in required_terms
    )
    expected_subquestions = expected.get("subquestions", ())
    actual_subquestions: tuple[SubQuestionAnswer, ...] = getattr(result, "subquestions", ())
    expected_count = len(expected_subquestions) if expected_subquestions else 1
    structured = bool(actual_subquestions)
    subquestion_count_hit = len(actual_subquestions) == expected_count
    if expected_outcome == "insufficient_evidence":
        grounded = outcome_hit and not citations and error is None
    else:
        grounded = (
            outcome_hit
            and complete_coverage
            and all(valid)
            and all(supported)
            and required_terms_pass
            and error is None
        )
    failures = []
    if error:
        failures.append("service_or_model_error")
    if not outcome_hit:
        failures.append("outcome")
    if not all(valid):
        failures.append("invalid_citation")
    if not all(supported):
        failures.append("citation_precision")
    if not complete_coverage:
        failures.append("evidence_coverage")
    if not required_terms_pass:
        failures.append("required_terms")
    if not subquestion_count_hit:
        failures.append("subquestion_count")
    return {
        "outcome_hit": outcome_hit,
        "citation_validity": citation_validity,
        "citation_precision": citation_precision,
        "evidence_recall": evidence_recall,
        "complete_evidence_coverage": complete_coverage,
        "required_terms_pass": required_terms_pass,
        "structured_subquestions": structured,
        "subquestion_count_hit": subquestion_count_hit,
        "automated_grounded_pass": grounded,
        "failure_types": failures,
    }


def run_case(case: dict[str, Any], repository: dict[str, Any], chat, embedding) -> dict[str, Any]:
    started = perf_counter()
    tracker = TrackedModels(chat, embedding)
    router_result = None
    result = None
    error = None
    revisions = 0
    events = []
    root = PROJECT_ROOT / repository["local_root"]
    try:
        router_result = route_question(case["input"]["question"], complete=tracker.complete)
        if router_result.plan.execution_route == "agent":
            agent_result = run_citation_agent(
                root,
                case["input"]["question"],
                complete=tracker.complete,
                retrieval_mode="auto",
                semantic_embed=tracker.embed,
                query_plan=router_result.plan,
            )
            result = agent_result.result
            result = type(
                "EvaluatedAgentResult",
                (),
                {
                    **result.__dict__,
                    "subquestions": agent_result.subquestions,
                },
            )()
            revisions = agent_result.revisions
            events = [asdict(item) for item in agent_result.events]
        else:
            result = auto_answer_repository(
                root,
                router_result=router_result,
                generate=tracker.generate,
                semantic_embed=tracker.embed
                if router_result.plan.execution_route != "insufficient"
                else None,
            )
            events = [asdict(item) for item in result.events]
    except Exception as caught:  # noqa: BLE001 - each case is an isolation boundary
        error = f"{type(caught).__name__}: {caught}"

    elapsed = (perf_counter() - started) * 1000
    metrics = _metrics(case, result, root, error)
    return {
        "case_id": case["id"],
        "repository_id": case["repository_id"],
        "language": case["language"],
        "category": case["category"],
        "scenario": case["scenario"],
        "expected_outcome": case["expected"]["outcome"],
        "router": (
            {
                "model": router_result.model,
                "used_fallback": router_result.used_fallback,
                "fallback_reason": router_result.fallback_reason,
                "plan": router_result.plan.to_dict(),
            }
            if router_result
            else None
        ),
        "result": _answer_payload(result) if result else None,
        "revisions": revisions,
        "events": events,
        "usage": asdict(tracker.usage),
        "elapsed_milliseconds": elapsed,
        "error": error,
        **metrics,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"case_count": 0}
    latencies = [item["elapsed_milliseconds"] for item in rows]
    routes = Counter(
        item["router"]["plan"]["execution_route"] if item["router"] else "router_error"
        for item in rows
    )
    return {
        "case_count": count,
        "outcome_accuracy": mean(item["outcome_hit"] for item in rows),
        "citation_validity": mean(item["citation_validity"] for item in rows),
        "citation_precision": mean(item["citation_precision"] for item in rows),
        "evidence_recall": mean(item["evidence_recall"] for item in rows),
        "complete_evidence_coverage": mean(item["complete_evidence_coverage"] for item in rows),
        "required_terms_case_rate": mean(item["required_terms_pass"] for item in rows),
        "structured_subquestion_rate": mean(item["structured_subquestions"] for item in rows),
        "subquestion_count_accuracy": mean(item["subquestion_count_hit"] for item in rows),
        "automated_grounded_pass": mean(item["automated_grounded_pass"] for item in rows),
        "error_count": sum(item["error"] is not None for item in rows),
        "router_fallback_count": sum(
            bool(item["router"] and item["router"]["used_fallback"]) for item in rows
        ),
        "route_distribution": dict(routes),
        "chat_calls": sum(item["usage"]["chat_calls"] for item in rows),
        "chat_input_tokens": sum(item["usage"]["chat_input_tokens"] for item in rows),
        "chat_output_tokens": sum(item["usage"]["chat_output_tokens"] for item in rows),
        "embedding_calls": sum(item["usage"]["embedding_calls"] for item in rows),
        "embedding_input_tokens": sum(item["usage"]["embedding_input_tokens"] for item in rows),
        "latency_ms": {
            "mean": mean(latencies),
            "p50": median(latencies),
            "p95": _percentile(latencies, 0.95),
        },
        "failure_types": dict(Counter(value for item in rows for value in item["failure_types"])),
    }


def summarize(rows: list[dict[str, Any]], repositories: list[str]) -> dict[str, Any]:
    by_repo = {
        repo: _aggregate([row for row in rows if row["repository_id"] == repo])
        for repo in repositories
    }
    nonempty_repo = [value for value in by_repo.values() if value.get("case_count")]
    macro_keys = (
        "outcome_accuracy",
        "citation_validity",
        "citation_precision",
        "evidence_recall",
        "complete_evidence_coverage",
        "automated_grounded_pass",
    )
    grouped: dict[str, dict[str, Any]] = {}
    for field in ("language", "category"):
        for value in sorted({row[field] for row in rows}):
            grouped[f"{field}:{value}"] = _aggregate([row for row in rows if row[field] == value])
    return {
        "micro": _aggregate(rows),
        "successful_calls_only": _aggregate([row for row in rows if not row["error"]]),
        "repository_equal_macro": {
            key: mean(item[key] for item in nonempty_repo) if nonempty_repo else 0.0
            for key in macro_keys
        },
        "by_repository": by_repo,
        "successful_calls_by_repository": {
            repo: _aggregate(
                [row for row in rows if row["repository_id"] == repo and not row["error"]]
            )
            for repo in repositories
        },
        "slices": grouped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repository", action="append", choices=("sample_repo", "httpx", "click", "requests")
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Rerun only selected cases whose latest stored result contains an error.",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    repositories = {item["id"]: item for item in document["repositories"]}
    selected_repositories = args.repository or list(repositories)
    cases = [
        item
        for item in document["cases"]
        if item["repository_id"] in selected_repositories
        and (not args.case_ids or item["id"] in args.case_ids)
    ]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    budget = {
        "run_id": args.run_id,
        "case_count": len(cases),
        "repositories": dict(Counter(item["repository_id"] for item in cases)),
        "chat_call_floor": len(cases) * 2,
        "embedding_index_builds": sum(
            item["expected"]["outcome"] != "insufficient_evidence" for item in cases
        ),
        "resume_policy": "existing case IDs in results.jsonl are skipped",
    }
    print(json.dumps(budget, ensure_ascii=False))
    if not args.confirm_run:
        return 0
    required = ("CODEINSIGHT_API_KEY", "CODEINSIGHT_MODEL", "CODEINSIGHT_EMBEDDING_MODEL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        parser.error("missing environment configuration: " + ", ".join(missing))

    run_dir = OUTPUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.jsonl"
    latest_rows: dict[str, dict[str, Any]] = {}
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest_rows[row["case_id"]] = row

    chat = OpenAIChatModel.from_environment()
    embedding = OpenAIEmbeddingModel.from_environment()
    pending = []
    for case in cases:
        previous = latest_rows.get(case["id"])
        if previous is None or (args.retry_errors and previous.get("error")):
            pending.append(case)
    for index, case in enumerate(pending, start=1):
        row = run_case(case, repositories[case["repository_id"]], chat, embedding)
        with result_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        latest_rows[row["case_id"]] = row
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(pending)}",
                    "case_id": case["id"],
                    "route": row["router"]["plan"]["execution_route"] if row["router"] else None,
                    "pass": row["automated_grounded_pass"],
                    "error": row["error"],
                    "elapsed_ms": round(row["elapsed_milliseconds"], 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    selected_rows = [latest_rows[case["id"]] for case in cases if case["id"] in latest_rows]
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "case_set": document["case_set"],
        "model": chat.model,
        "embedding_model": embedding.model,
        "selected_case_count": len(cases),
        "completed_case_count": len(selected_rows),
        "summary": summarize(selected_rows, selected_repositories),
        "notes": [
            (
                "The runner follows the current Auto Answer router decision; "
                "it does not force linear or Agent."
            ),
            (
                "Every case rebuilds the in-memory semantic index, matching "
                "current request-scoped product behavior."
            ),
            (
                "Metrics are deterministic and automated; no LLM judge or "
                "human-verified correctness is claimed."
            ),
            (
                "JSONL is append-only and resumable by case ID; one case failure "
                "does not stop later cases."
            ),
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": str(run_dir / "summary.json"), **payload["summary"]["micro"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
