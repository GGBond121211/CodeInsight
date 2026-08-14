"""Run the B00-B11 structured Answer/Critic ablation."""

from __future__ import annotations

import argparse
import json
import os
import random
from time import perf_counter
from typing import Any

from openai import OpenAI

from codeinsight.domain.answer import INSUFFICIENT_EVIDENCE, ModelCompletion
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.infrastructure.openai_chat import OpenAIChatModel

try:
    from .temporary_30_ablation import (
        FIXTURE_ROOT,
        MANIFEST_PATH,
        MAX_REVISIONS,
        SEED,
        StructuredCaseAnswer,
        changed_subquestion_ids,
        environment_fingerprint,
        git_state,
        parse_structured_answer,
        parse_structured_review,
        run_directory,
        selected_cases,
        sha256_file,
        write_json_new,
        write_jsonl_new,
    )
    from .temporary_30_ablation_prompts import (
        STRUCTURED_PROMPT_VERSION,
        STRUCTURED_REVIEW_VERSION,
        build_structured_draft_prompt,
        build_structured_review_prompt,
        build_structured_revision_prompt,
    )
except ImportError:  # Direct script execution.
    from temporary_30_ablation import (
        FIXTURE_ROOT,
        MANIFEST_PATH,
        MAX_REVISIONS,
        SEED,
        StructuredCaseAnswer,
        changed_subquestion_ids,
        environment_fingerprint,
        git_state,
        parse_structured_answer,
        parse_structured_review,
        run_directory,
        selected_cases,
        sha256_file,
        write_json_new,
        write_jsonl_new,
    )
    from temporary_30_ablation_prompts import (
        STRUCTURED_PROMPT_VERSION,
        STRUCTURED_REVIEW_VERSION,
        build_structured_draft_prompt,
        build_structured_review_prompt,
        build_structured_revision_prompt,
    )


def _covers(citation: dict, expected: dict) -> bool:
    return (
        citation["path"] == expected["path"]
        and citation["start_line"] <= expected["start_line"]
        and citation["end_line"] >= expected["end_line"]
    )


def _evidence_payload(results, evidence_ids) -> list[dict]:
    return [
        {
            "evidence_id": evidence_id,
            "path": result.chunk.relative_path,
            "start_line": result.chunk.start_line,
            "end_line": result.chunk.end_line,
            "rank": result.rank,
            "score": result.score,
            "retrieval_reason": result.retrieval_reason,
        }
        for result, evidence_id in zip(results, evidence_ids, strict=True)
    ]


class FrozenEvidenceSet:
    """Rebuild evidence text from the retrieval ablation's frozen A2/A5 Top-5."""

    def __init__(self, retrieval_path) -> None:
        payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        self.metadata = {
            "semantic_index_id": payload["semantic_index_id"],
            "embedding_model": payload["embedding_model"],
        }
        self._groups = {
            (row["case_id"], row["subquestion_id"]): row["groups"]
            for row in payload["subquestions"]
        }
        self._files = {
            path.relative_to(FIXTURE_ROOT).as_posix(): path.read_text(encoding="utf-8").splitlines()
            for path in FIXTURE_ROOT.rglob("*")
            if path.is_file()
        }

    def evidence(self, case_id: str, subquestion_id: str, hybrid: bool) -> tuple:
        group = "A5" if hybrid else "A2"
        frozen = self._groups[(case_id, subquestion_id)][group]["final_top5"]
        results = []
        for item in frozen:
            lines = self._files[item["path"]]
            chunk = SourceChunk(
                relative_path=item["path"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                text="\n".join(lines[item["start_line"] - 1 : item["end_line"]]),
            )
            results.append(
                RankedChunk(
                    chunk=chunk,
                    score=item["score"],
                    rank=item["rank"],
                    retrieval_reason=item["retrieval_reason"],
                )
            )
        return tuple(results)


def _complete(model: OpenAIChatModel, system: str, user: str) -> tuple[ModelCompletion, float]:
    started = perf_counter()
    completion = model.complete(system, user)
    return completion, (perf_counter() - started) * 1000


def _answer_metrics(
    case: dict,
    answer: StructuredCaseAnswer,
    evidence_groups: list[tuple[tuple, tuple[str, ...]]],
) -> dict:
    evidence_by_id = {
        evidence_id: {
            "path": result.chunk.relative_path,
            "start_line": result.chunk.start_line,
            "end_line": result.chunk.end_line,
        }
        for results, evidence_ids in evidence_groups
        for result, evidence_id in zip(results, evidence_ids, strict=True)
    }
    metrics = []
    for expected, actual in zip(case["subquestions"], answer.subquestions, strict=True):
        citations = [evidence_by_id[item] for item in actual.evidence_ids]
        expected_evidence = expected["expected_evidence"]
        evidence_hits = [
            any(_covers(citation, item) for citation in citations) for item in expected_evidence
        ]
        outcome_hit = actual.outcome == expected["expected_outcome"]
        if expected["expected_outcome"] == INSUFFICIENT_EVIDENCE:
            grounded = outcome_hit and not citations
            recall = None
        else:
            recall = sum(evidence_hits) / len(evidence_hits) if evidence_hits else 0.0
            grounded = outcome_hit and bool(evidence_hits) and all(evidence_hits)
        precision_hits = [
            any(_covers(citation, item) for item in expected_evidence) for citation in citations
        ]
        metrics.append(
            {
                "subquestion_id": expected["id"],
                "outcome_hit": outcome_hit,
                "citation_validity": 1.0,
                "citation_precision": (
                    sum(precision_hits) / len(precision_hits)
                    if precision_hits
                    else (1.0 if expected["expected_outcome"] == INSUFFICIENT_EVIDENCE else 0.0)
                ),
                "evidence_recall": recall,
                "automated_grounded_pass": grounded,
                "human_verified_complete_pass": None,
                "human_review_status": "not_human_reviewed",
            }
        )
    return {
        "subquestions": metrics,
        "outcome_accuracy": sum(item["outcome_hit"] for item in metrics) / len(metrics),
        "citation_precision": sum(item["citation_precision"] for item in metrics) / len(metrics),
        "evidence_recall": (
            sum(item["evidence_recall"] for item in metrics if item["evidence_recall"] is not None)
            / sum(item["evidence_recall"] is not None for item in metrics)
            if any(item["evidence_recall"] is not None for item in metrics)
            else None
        ),
        "automated_grounded_pass": all(item["automated_grounded_pass"] for item in metrics),
        "human_verified_complete_pass": None,
        "human_review_status": "not_human_reviewed",
    }


def _run_critic(
    model: OpenAIChatModel,
    case: dict,
    evidence_groups: list[tuple[tuple, tuple[str, ...]]],
    draft: StructuredCaseAnswer,
) -> dict:
    current = draft
    allowed = {
        subquestion["id"]: frozenset(evidence_ids)
        for subquestion, (_, evidence_ids) in zip(
            case["subquestions"], evidence_groups, strict=True
        )
    }
    expected_ids = tuple(item["id"] for item in case["subquestions"])
    snapshots = {"0": current.to_model_dict()}
    trajectory = []
    total_input = 0
    total_output = 0
    total_elapsed = 0.0
    stop_reason = "max_revisions"
    review_count = 0
    revise_count = 0
    for revision in range(MAX_REVISIONS + 1):
        review_prompt = build_structured_review_prompt(
            case["case_id"],
            case["original_question"],
            case["subquestions"],
            evidence_groups,
            current.to_model_dict(),
        )
        completion, elapsed = _complete(model, *review_prompt)
        review = parse_structured_review(
            completion.content,
            expected_subquestion_ids=expected_ids,
            allowed_evidence=allowed,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        review_count += 1
        total_input += review.input_tokens
        total_output += review.output_tokens
        total_elapsed += elapsed
        event: dict[str, Any] = {
            "revision": revision,
            "review": review.to_model_dict(),
            "review_input_tokens": review.input_tokens,
            "review_output_tokens": review.output_tokens,
            "review_elapsed_milliseconds": elapsed,
        }
        if all(item.verdict == "pass" for item in review.reviews):
            event["answer"] = current.to_model_dict()
            trajectory.append(event)
            stop_reason = "critic_pass"
            break
        if revision == MAX_REVISIONS:
            event["answer"] = current.to_model_dict()
            trajectory.append(event)
            break
        revision_prompt = build_structured_revision_prompt(
            case["case_id"],
            case["original_question"],
            case["subquestions"],
            evidence_groups,
            current.to_model_dict(),
            review.to_model_dict(),
        )
        completion, elapsed = _complete(model, *revision_prompt)
        revised = parse_structured_answer(
            completion.content,
            case_id=case["case_id"],
            expected_subquestion_ids=expected_ids,
            allowed_evidence=allowed,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        event.update(
            {
                "revised_answer": revised.to_model_dict(),
                "unauthorized_changed_subquestions": list(
                    changed_subquestion_ids(current, revised, review)
                ),
                "revise_input_tokens": revised.input_tokens,
                "revise_output_tokens": revised.output_tokens,
                "revise_elapsed_milliseconds": elapsed,
            }
        )
        trajectory.append(event)
        current = revised
        revise_count += 1
        total_input += revised.input_tokens
        total_output += revised.output_tokens
        total_elapsed += elapsed
        if revise_count in {1, 3, 5}:
            snapshots[str(revise_count)] = current.to_model_dict()
    for checkpoint in ("1", "3", "5"):
        snapshots.setdefault(checkpoint, current.to_model_dict())
    return {
        "final": current,
        "trajectory": trajectory,
        "snapshots": snapshots,
        "review_calls": review_count,
        "revise_calls": revise_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "elapsed_milliseconds": total_elapsed,
        "stop_reason": stop_reason,
    }


def run_case_condition(
    model: OpenAIChatModel,
    candidates: FrozenEvidenceSet,
    case: dict,
    *,
    repeat: int,
    hybrid: bool,
) -> tuple[dict, dict | None]:
    evidence_groups = []
    allowed: dict[str, frozenset[str]] = {}
    for subquestion in case["subquestions"]:
        results = candidates.evidence(case["case_id"], subquestion["id"], hybrid)
        evidence_ids = tuple(f"{subquestion['id']}E{index}" for index in range(1, len(results) + 1))
        evidence_groups.append((results, evidence_ids))
        allowed[subquestion["id"]] = frozenset(evidence_ids)
    expected_ids = tuple(item["id"] for item in case["subquestions"])
    prompt = build_structured_draft_prompt(
        case["case_id"],
        case["original_question"],
        case["subquestions"],
        evidence_groups,
    )
    completion, draft_elapsed = _complete(model, *prompt)
    draft = parse_structured_answer(
        completion.content,
        case_id=case["case_id"],
        expected_subquestion_ids=expected_ids,
        allowed_evidence=allowed,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
    evidence_payload = [
        {
            "subquestion_id": subquestion["id"],
            "question": subquestion["question"],
            "evidence": _evidence_payload(results, evidence_ids),
        }
        for subquestion, (results, evidence_ids) in zip(
            case["subquestions"], evidence_groups, strict=True
        )
    ]
    base = {
        "case_id": case["case_id"],
        "split": case["split"],
        "language": case["language"],
        "category": case["category"],
        "repeat": repeat,
        "hybrid": hybrid,
        "evidence_groups": evidence_payload,
        "draft": draft.to_model_dict(),
        "draft_raw_content": draft.raw_content,
        "draft_input_tokens": draft.input_tokens,
        "draft_output_tokens": draft.output_tokens,
        "draft_elapsed_milliseconds": draft_elapsed,
    }
    no_critic_group = "B10" if hybrid else "B00"
    critic_group = "B11" if hybrid else "B01"
    no_critic = {
        **base,
        "group": no_critic_group,
        "critic_enabled": False,
        "final": draft.to_model_dict(),
        "metrics": _answer_metrics(case, draft, evidence_groups),
        "model_calls": 1,
        "input_tokens": draft.input_tokens,
        "output_tokens": draft.output_tokens,
        "elapsed_milliseconds": draft_elapsed,
        "error": None,
    }
    try:
        critic = _run_critic(model, case, evidence_groups, draft)
        critic_result = {
            **base,
            "group": critic_group,
            "critic_enabled": True,
            "final": critic["final"].to_model_dict(),
            "metrics": _answer_metrics(case, critic["final"], evidence_groups),
            "model_calls": 1 + critic["review_calls"] + critic["revise_calls"],
            "review_calls": critic["review_calls"],
            "revise_calls": critic["revise_calls"],
            "input_tokens": draft.input_tokens + critic["input_tokens"],
            "output_tokens": draft.output_tokens + critic["output_tokens"],
            "elapsed_milliseconds": draft_elapsed + critic["elapsed_milliseconds"],
            "stop_reason": critic["stop_reason"],
            "snapshots": critic["snapshots"],
            "error": None,
        }
        trajectory = {
            "case_id": case["case_id"],
            "repeat": repeat,
            "group": critic_group,
            "events": critic["trajectory"],
            "stop_reason": critic["stop_reason"],
        }
    except Exception as error:  # noqa: BLE001 - preserve successful shared draft
        critic_result = {
            **base,
            "group": critic_group,
            "critic_enabled": True,
            "error": f"{type(error).__name__}: {error}",
            "metrics": {
                "automated_grounded_pass": False,
                "human_verified_complete_pass": None,
                "human_review_status": "not_human_reviewed",
            },
        }
        trajectory = None
    return (no_critic, critic_result), trajectory


def _error_rows(case: dict, repeat: int, hybrid: bool, error: Exception) -> list[dict]:
    groups = ("B10", "B11") if hybrid else ("B00", "B01")
    return [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "language": case["language"],
            "category": case["category"],
            "repeat": repeat,
            "group": group,
            "error": f"{type(error).__name__}: {error}",
            "metrics": {
                "automated_grounded_pass": False,
                "human_verified_complete_pass": None,
                "human_review_status": "not_human_reviewed",
            },
        }
        for group in groups
    ]


def build_answer_results(
    split: str, repetitions: int, model, retrieval_path
) -> tuple[list, list, dict]:
    cases = selected_cases(split)
    candidates = FrozenEvidenceSet(retrieval_path)
    tasks = [(case, repeat) for case in cases for repeat in range(1, repetitions + 1)]
    random.Random(SEED).shuffle(tasks)
    rows = []
    trajectories = []
    total_conditions = len(tasks) * 2
    completed_conditions = 0
    for case, repeat in tasks:
        for hybrid in (False, True):
            try:
                condition_rows, trajectory = run_case_condition(
                    model, candidates, case, repeat=repeat, hybrid=hybrid
                )
                rows.extend(condition_rows)
                if trajectory is not None:
                    trajectories.append(trajectory)
            except Exception as error:  # noqa: BLE001 - preserve condition error
                rows.extend(_error_rows(case, repeat, hybrid, error))
            completed_conditions += 1
            print(
                json.dumps(
                    {
                        "progress": f"{completed_conditions}/{total_conditions}",
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "condition": "hybrid" if hybrid else "sparse",
                        "errors_so_far": sum(item["error"] is not None for item in rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    error_count = sum(item["error"] is not None for item in rows)
    return (
        rows,
        trajectories,
        {
            "case_count": len(cases),
            "repetitions": repetitions,
            "row_count": len(rows),
            "errors": error_count,
            "error_rate": error_count / len(rows) if rows else 1.0,
            **candidates.metadata,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("diagnostic", "confirmation"), required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    cases = selected_cases(args.split)
    case_runs = len(cases) * args.repetitions
    budget = {
        "split": args.split,
        "cases": len(cases),
        "repetitions": args.repetitions,
        "case_runs": case_runs,
        "draft_calls": case_runs * 2,
        "minimum_chat_calls": case_runs * 4,
        "maximum_chat_calls": case_runs * 24,
        "embedding_index_batches": 0,
        "embedding_query_calls": 0,
        "evidence_source": "frozen retrieval-ablation A2/A5 Top-5",
        "environment": environment_fingerprint(),
    }
    print(json.dumps(budget, ensure_ascii=False))
    if args.dry_run or not args.confirm_run:
        return 0
    if not os.environ.get("CODEINSIGHT_EMBEDDING_MODEL"):
        parser.error("CODEINSIGHT_EMBEDDING_MODEL is required")
    output_dir = run_directory(args.run_id, args.split)
    retrieval_path = output_dir / "retrieval-ablation.json"
    if not retrieval_path.is_file():
        parser.error(f"run retrieval ablation first: {retrieval_path}")
    output_paths = (
        output_dir / "structured-subquestion-outputs.jsonl",
        output_dir / "critic-trajectories.jsonl",
        output_dir / "answer-run-manifest.json",
    )
    if any(path.exists() for path in output_paths):
        parser.error("one or more answer ablation outputs already exist")
    api_key = os.environ.get("CODEINSIGHT_API_KEY")
    model_name = os.environ.get("CODEINSIGHT_MODEL")
    if not api_key or not model_name:
        parser.error("CODEINSIGHT_API_KEY and CODEINSIGHT_MODEL are required")
    model = OpenAIChatModel(
        client=OpenAI(
            api_key=api_key,
            base_url=os.environ.get("CODEINSIGHT_BASE_URL") or None,
            timeout=120.0,
            max_retries=0,
        ),
        model=model_name,
        response_format={"type": "json_object"},
    )
    started = perf_counter()
    rows, trajectories, summary = build_answer_results(
        args.split, args.repetitions, model, retrieval_path
    )
    summary["elapsed_milliseconds"] = (perf_counter() - started) * 1000
    write_jsonl_new(output_paths[0], rows)
    write_jsonl_new(output_paths[1], trajectories)
    write_json_new(
        output_paths[2],
        {
            "schema_version": 1,
            "split": args.split,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "prompt_version": STRUCTURED_PROMPT_VERSION,
            "review_version": STRUCTURED_REVIEW_VERSION,
            "max_revisions": MAX_REVISIONS,
            "environment": environment_fingerprint(),
            "git": git_state(),
            "summary": summary,
        },
    )
    print(
        json.dumps(
            {"outputs": [str(path) for path in output_paths], "summary": summary},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
