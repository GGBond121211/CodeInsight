"""Compare the current lexical, BM25, and hybrid retrieval modes."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from codeinsight.application.search_repository import search_repository
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex
from codeinsight.domain.source import SourceChunk
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.semantic import build_semantic_index, search_chunks_semantic

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "multilingual_cases.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"

LIMIT = 5
DIAGNOSTIC_LIMIT = 20
CHUNK_MAX_LINES = 80
RETRIEVAL_MODES = ("lexical", "bm25", "hybrid")
STRUCTURAL_CATEGORIES = {"boundary_behavior", "cross_file_trace", "data_flow"}
_CODE_IDENTIFIER_PATTERN = re.compile(r"`[^`]+`")


def evidence_items(case: dict) -> tuple[dict, ...]:
    """Return unique expected evidence from a case and its answerable subquestions."""
    expected = case.get("expected", {})
    candidates = list(expected.get("evidence", ()))
    if not candidates:
        for subquestion in expected.get("subquestions", ()):
            for item in subquestion.get("evidence", ()):
                candidates.append(item)
    unique: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for item in candidates:
        key = (item["path"], item["start_line"], item["end_line"])
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return tuple(unique)


def case_groups(case: dict) -> tuple[str, ...]:
    """Derive stable, input-only groups for aggregate diagnostics."""
    groups = [
        "all",
        f"language:{case['language']}",
        f"category:{case['category']}",
    ]
    if case.get("source_set"):
        groups.append(f"source:{case['source_set']}")
    if case.get("language") == "zh-en":
        groups.append("mixed_language")
    challenge_features = set(case.get("challenge_features", ()))
    if (
        case.get("noise_profile")
        or case.get("category") == "noisy_query"
        or "typo_noise" in challenge_features
        or "log_fragment" in challenge_features
    ):
        groups.append("typo_or_noise")
    if case.get("category") == "multi_intent":
        groups.append("multi_question")
    if case.get("category") == "semantic_paraphrase":
        groups.append("semantic_mismatch")
    if _CODE_IDENTIFIER_PATTERN.search(case["input"]["question"]):
        groups.append("exact_identifier")
    return tuple(groups)


def covers(chunk: SourceChunk, evidence: dict) -> bool:
    return (
        chunk.relative_path == evidence["path"]
        and chunk.start_line <= evidence["start_line"]
        and chunk.end_line >= evidence["end_line"]
    )


def coverage_rank(results: Sequence[RankedChunk], evidence: tuple[dict, ...]) -> int | None:
    covered = [False] * len(evidence)
    for result in results:
        for index, item in enumerate(evidence):
            covered[index] = covered[index] or covers(result.chunk, item)
        if all(covered):
            return result.rank
    return None


def evidence_recall(results: Sequence[RankedChunk], evidence: tuple[dict, ...]) -> bool:
    for item in evidence:
        item_is_covered = False
        for result in results:
            if covers(result.chunk, item):
                item_is_covered = True
                break
        if not item_is_covered:
            return False
    return True


def evidence_lines(evidence: tuple[dict, ...]) -> set[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for item in evidence:
        for line_number in range(item["start_line"], item["end_line"] + 1):
            lines.add((item["path"], line_number))
    return lines


def chunk_lines(chunk: SourceChunk) -> set[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for line_number in range(chunk.start_line, chunk.end_line + 1):
        lines.add((chunk.relative_path, line_number))
    return lines


def is_valid_chunk(chunk: SourceChunk) -> bool:
    if not chunk.relative_path or chunk.start_line < 1 or chunk.end_line < chunk.start_line:
        return False
    target = FIXTURE_ROOT / Path(chunk.relative_path)
    if not target.is_file():
        return False
    return chunk.end_line <= len(target.read_text(encoding="utf-8").splitlines())


def result_payload(result: RankedChunk) -> dict:
    chunk = result.chunk
    return {
        "rank": result.rank,
        "score": result.score,
        "relative_path": chunk.relative_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol_path": chunk.symbol_path,
        "retrieval_reason": result.retrieval_reason,
    }


def classify_failure(
    case: dict,
    retrieval_mode: str,
    top5: Sequence[RankedChunk],
    top20: Sequence[RankedChunk],
    evidence: tuple[dict, ...],
) -> tuple[str | None, str | None]:
    """Classify a failed top-5 result without using an LLM judge."""
    if not evidence:
        return "insufficient_evidence", None
    if evidence_recall(top5, evidence):
        return None, None
    if evidence_recall(top20, evidence):
        return "top_k_capacity", "evidence_is_present_by_rank_20"

    expected_paths = set()
    for item in evidence:
        expected_paths.add(item["path"])
    returned_paths = set()
    for result in top20:
        returned_paths.add(result.chunk.relative_path)
    has_expected_path = bool(expected_paths & returned_paths)
    category = case["category"]
    if category == "multi_intent":
        return "decomposition_failure", "mixed_question_not_split_before_retrieval"
    if category in STRUCTURAL_CATEGORIES and retrieval_mode == "bm25":
        return "structural_mismatch", "required_cross_file_or_boundary_evidence_not_recovered"
    if has_expected_path:
        return "chunk_mismatch", "expected_path_returned_without_required_line_span"
    if category == "semantic_paraphrase":
        return "lexical_mismatch", "semantic_mismatch_case"
    return "lexical_mismatch", "no_expected_path_in_top_20"


def _search_fn_default(
    root: str | Path,
    question: str,
    *,
    limit: int,
    chunk_max_lines: int,
    retrieval_mode: str,
) -> tuple[RankedChunk, ...]:
    return search_repository(
        root,
        question,
        limit=limit,
        chunk_max_lines=chunk_max_lines,
        retrieval_mode=retrieval_mode,
    )


def _search_fn_with_embedding(
    embed: Callable[[Sequence[str]], EmbeddingBatch],
) -> Callable[..., tuple[RankedChunk, ...]]:
    """Build an evaluation search function with one reusable semantic index."""
    semantic_indexes: dict[tuple[str, int], SemanticIndex] = {}

    def search_fn(
        root: str | Path,
        question: str,
        *,
        limit: int,
        chunk_max_lines: int,
        retrieval_mode: str,
    ) -> tuple[RankedChunk, ...]:
        if retrieval_mode == "semantic":
            key = (str(Path(root).resolve()), chunk_max_lines)
            index = semantic_indexes.get(key)
            if index is None:
                scan_result = scan_repository(root)
                chunks = chunk_scan_result(scan_result, max_lines=chunk_max_lines)
                index = build_semantic_index(chunks, embed)
                semantic_indexes[key] = index
            return search_chunks_semantic(question, index, embed, limit=limit)
        if retrieval_mode == "hybrid":
            return search_repository(
                root,
                question,
                limit=limit,
                chunk_max_lines=chunk_max_lines,
                retrieval_mode="hybrid",
                semantic_embed=embed,
            )
        return _search_fn_default(
            root,
            question,
            limit=limit,
            chunk_max_lines=chunk_max_lines,
            retrieval_mode=retrieval_mode,
        )

    return search_fn


def build_case_payload(
    case: dict,
    retrieval_mode: str,
    *,
    search_fn: Callable[..., tuple[RankedChunk, ...]] = _search_fn_default,
) -> dict:
    evidence = evidence_items(case)
    groups = case_groups(case)
    if not evidence:
        return {
            "case_id": case["id"],
            "language": case["language"],
            "category": case["category"],
            "groups": groups,
            "applicable": False,
            "evidence_scope": "none",
            "expected_evidence_count": 0,
            "failure_class": "insufficient_evidence",
            "failure_detail": None,
            "results": [],
        }

    question = case["input"]["question"]
    top5 = search_fn(
        FIXTURE_ROOT,
        question,
        limit=LIMIT,
        chunk_max_lines=CHUNK_MAX_LINES,
        retrieval_mode=retrieval_mode,
    )
    top20 = search_fn(
        FIXTURE_ROOT,
        question,
        limit=DIAGNOSTIC_LIMIT,
        chunk_max_lines=CHUNK_MAX_LINES,
        retrieval_mode=retrieval_mode,
    )
    expected_lines = evidence_lines(evidence)
    returned_lines: set[tuple[str, int]] = set()
    context_lines = 0
    for item in top5:
        returned_lines.update(chunk_lines(item.chunk))
        context_lines += item.chunk.end_line - item.chunk.start_line + 1
    covered_lines = expected_lines & returned_lines
    covering_rank = coverage_rank(top5, evidence)
    failure_class, failure_detail = classify_failure(
        case,
        retrieval_mode,
        top5,
        top20,
        evidence,
    )
    retrieval_reasons = []
    valid_count = 0
    for item in top5:
        retrieval_reasons.append(item.retrieval_reason)
        if is_valid_chunk(item.chunk):
            valid_count += 1
    reason_counts = dict(sorted(Counter(retrieval_reasons).items()))
    top1_hit = False
    if top5:
        top1_hit = True
        for item in evidence:
            if not covers(top5[0].chunk, item):
                top1_hit = False
                break
    result_payloads = []
    for item in top5:
        result_payloads.append(result_payload(item))
    return {
        "case_id": case["id"],
        "source_set": case.get("source_set"),
        "language": case["language"],
        "category": case["category"],
        "groups": groups,
        "applicable": True,
        "evidence_scope": "top_level"
        if case.get("expected", {}).get("evidence")
        else "subquestions",
        "expected_evidence_count": len(evidence),
        "top1_hit": top1_hit,
        "coverage_rank_at_5": covering_rank,
        "recall_at_5": evidence_recall(top5, evidence),
        "valid_evidence_rate_at_5": valid_count / len(top5) if top5 else 0.0,
        "context_lines_at_5": context_lines,
        "evidence_line_recall_at_5": len(covered_lines) / len(expected_lines),
        "evidence_density_at_5": len(covered_lines) / context_lines if context_lines else 0.0,
        "retrieval_reason_counts": reason_counts,
        "failure_class": failure_class,
        "failure_detail": failure_detail,
        "results": result_payloads,
    }


def aggregate_case_payloads(case_payloads: Sequence[dict]) -> dict:
    applicable = []
    for item in case_payloads:
        if item["applicable"]:
            applicable.append(item)
    count = len(applicable)
    returned_total = 0
    for item in applicable:
        returned_total += len(item["results"])
    reason_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    top1_hits = 0
    reciprocal_ranks = []
    recall_values = []
    context_line_values = []
    evidence_line_recall_values = []
    evidence_density_values = []
    valid_result_count = 0
    for item in applicable:
        reason_counts.update(item["retrieval_reason_counts"])
        if item["failure_class"]:
            failure_counts[item["failure_class"]] += 1
        if item["top1_hit"]:
            top1_hits += 1
        if item["coverage_rank_at_5"]:
            reciprocal_ranks.append(1.0 / item["coverage_rank_at_5"])
        else:
            reciprocal_ranks.append(0.0)
        recall_values.append(item["recall_at_5"])
        context_line_values.append(item["context_lines_at_5"])
        evidence_line_recall_values.append(item["evidence_line_recall_at_5"])
        evidence_density_values.append(item["evidence_density_at_5"])
        for result in item["results"]:
            chunk = SourceChunk(
                result["relative_path"],
                result["start_line"],
                result["end_line"],
                "",
                result["symbol_path"],
            )
            if is_valid_chunk(chunk):
                valid_result_count += 1
    return {
        "case_count": len(case_payloads),
        "applicable_case_count": count,
        "top1": top1_hits / count if count else 0.0,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / count if count else 0.0,
        "recall_at_5": sum(recall_values) / count if count else 0.0,
        "valid_evidence_rate": (
            valid_result_count / returned_total
            if returned_total
            else 0.0
        ),
        "mean_context_lines_at_5": sum(context_line_values) / count if count else 0.0,
        "mean_evidence_line_recall_at_5": sum(evidence_line_recall_values) / count
        if count
        else 0.0,
        "mean_evidence_density_at_5": sum(evidence_density_values) / count if count else 0.0,
        "retrieval_reason_counts": dict(sorted(reason_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def grouped_metrics(case_payloads: Sequence[dict]) -> dict:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for item in case_payloads:
        for group in item["groups"]:
            grouped[group].append(item)
    metrics = {}
    for group, items in sorted(grouped.items()):
        metrics[group] = aggregate_case_payloads(items)
    return metrics


def build_mode_payload(
    cases: Sequence[dict],
    retrieval_mode: str,
    *,
    search_fn: Callable[..., tuple[RankedChunk, ...]] = _search_fn_default,
) -> dict:
    case_payloads = []
    for case in cases:
        case_payloads.append(build_case_payload(case, retrieval_mode, search_fn=search_fn))
    return {
        "retrieval_mode": retrieval_mode,
        "limit": LIMIT,
        "diagnostic_limit": DIAGNOSTIC_LIMIT,
        "chunk_max_lines": CHUNK_MAX_LINES,
        "strict_metrics": aggregate_case_payloads(case_payloads),
        "groups": grouped_metrics(case_payloads),
        "cases": case_payloads,
    }


def build_comparison(
    document: dict,
    *,
    search_fn: Callable[..., tuple[RankedChunk, ...]] = _search_fn_default,
    retrieval_modes: Sequence[str] = RETRIEVAL_MODES,
) -> dict:
    cases = document["cases"]
    modes = {}
    for mode in retrieval_modes:
        modes[mode] = build_mode_payload(cases, mode, search_fn=search_fn)
    applicable_case_count = 0
    for case in cases:
        if evidence_items(case):
            applicable_case_count += 1
    return {
        "schema_version": 1,
        "case_set": document["case_set"],
        "fixture": document["fixture"],
        "case_count": len(cases),
        "applicable_case_count": applicable_case_count,
        "retrieval_modes": list(retrieval_modes),
        "evaluation_contract": {
            "top_k": LIMIT,
            "diagnostic_top_k": DIAGNOSTIC_LIMIT,
            "chunk_max_lines": CHUNK_MAX_LINES,
            "model_calls": 0,
            "embedding_calls": 0
            if not set(retrieval_modes) & {"semantic", "hybrid"}
            else "recorded_by_provider",
            "grouping": (
                "language, category, mixed_language, typo_or_noise, multi_question, "
                "semantic_mismatch, exact_identifier"
            ),
        },
        "modes": modes,
        "notes": [
            "All metrics are deterministic and use repository path/line evidence; "
            "no LLM judge was used.",
            "Top-20 is a diagnostic only and does not change the Top-5 acceptance metric.",
            "Cases without expected evidence are recorded as insufficient_evidence and "
            "excluded from retrieval scores.",
            "exact_identifier is derived from user-visible backtick code spans; no expected "
            "answer text is used for routing.",
        ],
        "decision": (
            "semantic_modes_enabled_for_explicit_embedding_config"
            if set(retrieval_modes) & {"semantic", "hybrid"}
            else "pending_task_11_3_gate"
        ),
    }


def main() -> int:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if os.environ.get("CODEINSIGHT_EMBEDDING_MODEL"):
        embedding_model = OpenAIEmbeddingModel.from_environment()
        retrieval_modes = RETRIEVAL_MODES + ("semantic", "hybrid")
        search_fn = _search_fn_with_embedding(embedding_model.embed)
        payload = build_comparison(
            document,
            search_fn=search_fn,
            retrieval_modes=retrieval_modes,
        )
    else:
        payload = build_comparison(document)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "multilingual-retrieval-comparison.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
