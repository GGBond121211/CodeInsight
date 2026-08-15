"""Shared contracts and utilities for the Temporary-30 ablation experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codeinsight.domain.answer import ANSWERED, INSUFFICIENT_EVIDENCE
from codeinsight.domain.errors import ModelResponseError

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
EVAL_ROOT = BACKEND_ROOT / "tests" / "evals"
CASES_PATH = EVAL_ROOT / "multilingual_cases_temporary_30.json"
MANIFEST_PATH = EVAL_ROOT / "temporary_30_ablation_manifest.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evals" / "temporary-30-ablation"
SEED = 20260812
MAX_REVISIONS = 5
RETRIEVAL_GROUPS = ("A0", "A1", "A2", "A3", "A4", "A5")
ANSWER_GROUPS = ("B00", "B01", "B10", "B11")


@dataclass(frozen=True)
class StructuredSubquestionAnswer:
    subquestion_id: str
    outcome: str
    answer: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subquestion_id": self.subquestion_id,
            "outcome": self.outcome,
            "answer": self.answer,
            "citations": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class StructuredCaseAnswer:
    case_id: str
    subquestions: tuple[StructuredSubquestionAnswer, ...]
    model: str
    input_tokens: int
    output_tokens: int
    raw_content: str

    def to_model_dict(self) -> dict[str, Any]:
        subquestion_outputs = []
        for item in self.subquestions:
            subquestion_outputs.append(item.to_dict())
        return {
            "case_id": self.case_id,
            "subquestion_outputs": subquestion_outputs,
        }


@dataclass(frozen=True)
class StructuredSubquestionReview:
    subquestion_id: str
    verdict: str
    supported_evidence_ids: tuple[str, ...]
    feedback: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "subquestion_id": self.subquestion_id,
            "verdict": self.verdict,
            "supported_citations": list(self.supported_evidence_ids),
            "feedback": self.feedback,
        }


@dataclass(frozen=True)
class StructuredCaseReview:
    reviews: tuple[StructuredSubquestionReview, ...]
    model: str
    input_tokens: int
    output_tokens: int
    raw_content: str

    def to_model_dict(self) -> dict[str, Any]:
        subquestion_reviews = []
        for item in self.reviews:
            subquestion_reviews.append(item.to_dict())
        return {"subquestion_reviews": subquestion_reviews}


def _decode_json_object(content: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ModelResponseError("structured evaluation response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ModelResponseError("structured evaluation response must be a JSON object")
    return payload


def _validate_exact_ids(actual: Sequence[str], expected: Sequence[str], label: str) -> None:
    if len(actual) != len(set(actual)):
        raise ModelResponseError(f"{label} contains duplicate subquestion IDs")
    if tuple(actual) != tuple(expected):
        raise ModelResponseError(f"{label} subquestion IDs must exactly match the manifest order")


def parse_structured_answer(
    content: str,
    *,
    case_id: str,
    expected_subquestion_ids: Sequence[str],
    allowed_evidence: dict[str, frozenset[str]],
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> StructuredCaseAnswer:
    payload = _decode_json_object(content)
    if frozenset(payload) != frozenset({"case_id", "subquestion_outputs"}):
        raise ModelResponseError(
            "structured answer must contain only case_id and subquestion_outputs"
        )
    if payload["case_id"] != case_id:
        raise ModelResponseError("structured answer case_id does not match")
    outputs = payload["subquestion_outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise ModelResponseError("subquestion_outputs must be a non-empty list")
    actual_ids = []
    for item in outputs:
        if isinstance(item, dict):
            actual_ids.append(item.get("subquestion_id"))
    all_ids_are_strings = True
    for item in actual_ids:
        if not isinstance(item, str):
            all_ids_are_strings = False
            break
    if len(actual_ids) != len(outputs) or not all_ids_are_strings:
        raise ModelResponseError("every structured answer item needs a subquestion_id")
    _validate_exact_ids(actual_ids, expected_subquestion_ids, "structured answer")

    parsed: list[StructuredSubquestionAnswer] = []
    for item in outputs:
        if frozenset(item) != frozenset({"subquestion_id", "outcome", "answer", "citations"}):
            raise ModelResponseError("structured answer item has unsupported fields")
        subquestion_id = item["subquestion_id"]
        outcome = item["outcome"]
        answer = item["answer"]
        citations = item["citations"]
        if outcome not in {ANSWERED, INSUFFICIENT_EVIDENCE}:
            raise ModelResponseError("structured answer has an unsupported outcome")
        if not isinstance(answer, str) or not answer.strip():
            raise ModelResponseError("structured answer text must be non-empty")
        citations_are_strings = True
        if isinstance(citations, list):
            for value in citations:
                if not isinstance(value, str):
                    citations_are_strings = False
                    break
        else:
            citations_are_strings = False
        if not citations_are_strings:
            raise ModelResponseError("structured answer citations must be evidence IDs")
        citations = list(dict.fromkeys(citations))
        unknown = []
        for value in citations:
            if value not in allowed_evidence[subquestion_id]:
                unknown.append(value)
        if unknown:
            raise ModelResponseError(
                f"{subquestion_id} used evidence owned by another group: {unknown[0]}"
            )
        if outcome == ANSWERED and not citations:
            raise ModelResponseError("answered subquestion must cite evidence")
        if outcome == INSUFFICIENT_EVIDENCE and citations:
            raise ModelResponseError("insufficient subquestion cannot cite evidence")
        parsed.append(
            StructuredSubquestionAnswer(
                subquestion_id=subquestion_id,
                outcome=outcome,
                answer=answer.strip(),
                evidence_ids=tuple(citations),
            )
        )
    return StructuredCaseAnswer(
        case_id=case_id,
        subquestions=tuple(parsed),
        model=model,
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        raw_content=content,
    )


def parse_structured_review(
    content: str,
    *,
    expected_subquestion_ids: Sequence[str],
    allowed_evidence: dict[str, frozenset[str]],
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> StructuredCaseReview:
    payload = _decode_json_object(content)
    if frozenset(payload) != frozenset({"subquestion_reviews"}):
        raise ModelResponseError("structured review must contain only subquestion_reviews")
    reviews = payload["subquestion_reviews"]
    if not isinstance(reviews, list) or not reviews:
        raise ModelResponseError("subquestion_reviews must be a non-empty list")
    actual_ids = []
    for item in reviews:
        if isinstance(item, dict):
            actual_ids.append(item.get("subquestion_id"))
    all_ids_are_strings = True
    for item in actual_ids:
        if not isinstance(item, str):
            all_ids_are_strings = False
            break
    if len(actual_ids) != len(reviews) or not all_ids_are_strings:
        raise ModelResponseError("every structured review item needs a subquestion_id")
    _validate_exact_ids(actual_ids, expected_subquestion_ids, "structured review")

    parsed: list[StructuredSubquestionReview] = []
    for item in reviews:
        if frozenset(item) != frozenset(
            {"subquestion_id", "verdict", "supported_citations", "feedback"}
        ):
            raise ModelResponseError("structured review item has unsupported fields")
        subquestion_id = item["subquestion_id"]
        verdict = item["verdict"]
        citations = item["supported_citations"]
        feedback = item["feedback"]
        if verdict not in {"pass", "revise"}:
            raise ModelResponseError("structured review verdict must be pass or revise")
        citations_are_strings = True
        if isinstance(citations, list):
            for value in citations:
                if not isinstance(value, str):
                    citations_are_strings = False
                    break
        else:
            citations_are_strings = False
        if not citations_are_strings:
            raise ModelResponseError("supported_citations must be evidence IDs")
        citations = list(dict.fromkeys(citations))
        unknown = []
        for value in citations:
            if value not in allowed_evidence[subquestion_id]:
                unknown.append(value)
        if unknown:
            raise ModelResponseError(
                f"{subquestion_id} review used evidence owned by another group: {unknown[0]}"
            )
        if not isinstance(feedback, str) or not feedback.strip():
            raise ModelResponseError("structured review feedback must be non-empty")
        parsed.append(
            StructuredSubquestionReview(
                subquestion_id=subquestion_id,
                verdict=verdict,
                supported_evidence_ids=tuple(citations),
                feedback=feedback.strip(),
            )
        )
    return StructuredCaseReview(
        reviews=tuple(parsed),
        model=model,
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        raw_content=content,
    )


def changed_subquestion_ids(
    before: StructuredCaseAnswer,
    after: StructuredCaseAnswer,
    review: StructuredCaseReview,
) -> tuple[str, ...]:
    allowed = set()
    for item in review.reviews:
        if item.verdict == "revise":
            allowed.add(item.subquestion_id)
    before_by_id = {}
    for item in before.subquestions:
        before_by_id[item.subquestion_id] = item
    changed_ids = []
    for item in after.subquestions:
        if item != before_by_id[item.subquestion_id] and item.subquestion_id not in allowed:
            changed_ids.append(item.subquestion_id)
    return tuple(changed_ids)


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def selected_cases(split: str) -> list[dict[str, Any]]:
    if split not in {"diagnostic", "confirmation"}:
        raise ValueError("split must be diagnostic or confirmation")
    selected = []
    for case in load_manifest()["cases"]:
        if case["split"] == split:
            selected.append(case)
    return selected


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_directory(run_id: str, split: str | None = None) -> Path:
    root = OUTPUT_ROOT / run_id
    return root / split if split else root


def ensure_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_new(path: Path, payload: Any) -> None:
    ensure_new_file(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_new_file(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def environment_fingerprint() -> dict[str, Any]:
    configured_variable_names = []
    variable_names = (
        "CODEINSIGHT_API_KEY",
        "CODEINSIGHT_BASE_URL",
        "CODEINSIGHT_MODEL",
        "CODEINSIGHT_EMBEDDING_API_KEY",
        "CODEINSIGHT_EMBEDDING_BASE_URL",
        "CODEINSIGHT_EMBEDDING_MODEL",
    )
    for key in variable_names:
        if os.environ.get(key):
            configured_variable_names.append(key)
    return {
        "chat_model": os.environ.get("CODEINSIGHT_MODEL"),
        "embedding_model": os.environ.get("CODEINSIGHT_EMBEDDING_MODEL"),
        "configured_variable_names": sorted(configured_variable_names),
    }


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def case_cluster_bootstrap(
    paired_case_values: dict[str, Sequence[float]],
    *,
    iterations: int = 10_000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bootstrap case means; all repeated observations travel with their case."""
    if not paired_case_values:
        return {"mean": None, "ci95": [None, None], "case_count": 0}
    case_ids = sorted(paired_case_values)
    case_means = {}
    for case_id, values in paired_case_values.items():
        if values:
            case_means[case_id] = sum(values) / len(values)
    if len(case_means) != len(case_ids):
        raise ValueError("every bootstrap case must contain at least one paired value")
    observed = sum(case_means.values()) / len(case_means)
    generator = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = []
        for _ in case_ids:
            selected.append(generator.choice(case_ids))
        selected_total = 0.0
        for case_id in selected:
            selected_total += case_means[case_id]
        samples.append(selected_total / len(selected))
    samples.sort()
    low = samples[max(0, math.ceil(0.025 * iterations) - 1)]
    high = samples[max(0, math.ceil(0.975 * iterations) - 1)]
    return {"mean": observed, "ci95": [low, high], "case_count": len(case_ids)}


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired binary samples must have equal lengths")
    wins = 0
    losses = 0
    for before, after in zip(left, right, strict=True):
        if not before and after:
            wins += 1
        if before and not after:
            losses += 1
    ties = len(left) - wins - losses
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = 0
        for k in range(0, min(wins, losses) + 1):
            tail += math.comb(discordant, k)
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {"wins": wins, "losses": losses, "ties": ties, "p_value": p_value}


def majority(values: Sequence[bool]) -> bool:
    if not values or len(values) % 2 == 0:
        raise ValueError("majority requires a non-empty odd number of values")
    return sum(values) > len(values) // 2
