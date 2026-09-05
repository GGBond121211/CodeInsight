"""Run the deterministic EXP-001 chunk-size/overlap comparison.

The runner reuses the frozen Master-200 questions and archived Router
subquestions. It makes no Chat calls; the only external calls are the real
configured Embedding requests needed to build and query each semantic index.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex
from codeinsight.domain.source import SourceChunk
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.bm25 import search_chunks_bm25
from codeinsight.retrieval.hybrid import rerank_ranked_chunks
from codeinsight.retrieval.semantic import build_semantic_index, search_chunks_semantic
from codeinsight.retrieval.vector_store import VectorStore

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
CASES_PATH = Path(__file__).with_name("master_200_cases.json")
ARCHIVED_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evals"
    / "master-200"
    / "20260813T-master200-final-v2"
    / "results.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "EXP-001"

CHUNK_PROFILES = tuple(
    {"max_lines": max_lines, "overlap_ratio": overlap_ratio}
    for max_lines in (40, 80, 120)
    for overlap_ratio in (0.0, 0.10, 0.20)
)
SOURCE_LIMIT = 20
FINAL_LIMIT = 5
SEMANTIC_MIN_SCORE = 0.20


class UsageTracker:
    """Track provider usage without exposing credentials or response bodies."""

    def __init__(
        self,
        model: OpenAIEmbeddingModel,
        *,
        query_cache: dict[str, EmbeddingBatch],
    ) -> None:
        self._model = model
        self._query_cache = query_cache
        self.calls = 0
        self.input_tokens = 0
        self.usage_available = True

    def embed(self, texts: tuple[str, ...] | list[str]) -> EmbeddingBatch:
        if len(texts) == 1 and texts[0] in self._query_cache:
            return self._query_cache[texts[0]]
        self.calls += math.ceil(len(texts) / 16)
        batch = self._model.embed(texts)
        if batch.input_tokens is None:
            self.usage_available = False
        else:
            self.input_tokens += batch.input_tokens
        if len(texts) == 1:
            self._query_cache[texts[0]] = batch
        return batch


def _key(result: RankedChunk) -> tuple[str, int, int]:
    chunk = result.chunk
    return chunk.relative_path, chunk.start_line, chunk.end_line


def _unique_evidence(case: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    expected = case.get("expected", {})
    values = list(expected.get("evidence", ()))
    for subquestion in expected.get("subquestions", ()):
        values.extend(subquestion.get("evidence", ()))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in values:
        key = (item["path"], item["start_line"], item["end_line"])
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return tuple(unique)


def _queries(case: dict[str, Any], archived: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    row = archived.get(case["id"])
    if row is None:
        raise RuntimeError(f"固定查询集缺少案例：{case['id']}")
    values = []
    for item in row.get("router", {}).get("plan", {}).get("subquestions", ()):
        question = item.get("question")
        if isinstance(question, str) and question.strip():
            values.append(question)
    if values:
        return tuple(values)
    question = case.get("input", {}).get("question")
    if isinstance(question, str) and question.strip():
        return (question,)
    raise RuntimeError(f"案例没有可执行查询：{case['id']}")


def _covers(chunk: SourceChunk, evidence: dict[str, Any]) -> bool:
    return (
        chunk.relative_path == evidence["path"]
        and chunk.start_line <= evidence["start_line"]
        and chunk.end_line >= evidence["end_line"]
    )


def _expected_lines(evidence: tuple[dict[str, Any], ...]) -> set[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for item in evidence:
        for line_number in range(item["start_line"], item["end_line"] + 1):
            lines.add((item["path"], line_number))
    return lines


def _chunk_lines(chunk: SourceChunk) -> set[tuple[str, int]]:
    return {
        (chunk.relative_path, line_number)
        for line_number in range(chunk.start_line, chunk.end_line + 1)
    }


def _valid_chunk(chunk: SourceChunk, root: Path, line_counts: dict[str, int]) -> bool:
    return (
        bool(chunk.relative_path)
        and chunk.start_line >= 1
        and chunk.end_line >= chunk.start_line
        and chunk.end_line <= line_counts.get(chunk.relative_path, 0)
        and (root / chunk.relative_path).is_file()
    )


def _build_index(
    root: Path,
    *,
    max_lines: int,
    overlap_ratio: float,
    tracker: UsageTracker,
) -> tuple[tuple[SourceChunk, ...], SemanticIndex, dict[str, int], dict[str, Any]]:
    scan_result = scan_repository(root)
    chunks = chunk_scan_result(
        scan_result,
        max_lines=max_lines,
        overlap_ratio=overlap_ratio,
    )
    line_counts: dict[str, int] = {}
    for source in scan_result.files:
        normalized = source.text.replace("\r\n", "\n").replace("\r", "\n")
        line_counts[source.relative_path] = len(normalized.splitlines())
    before_tokens = tracker.input_tokens
    before_calls = tracker.calls
    started = time.perf_counter()
    index = build_semantic_index(chunks, tracker.embed)
    elapsed = time.perf_counter() - started
    serialized_size = len(pickle.dumps(index, protocol=5))
    return (
        chunks,
        index,
        line_counts,
        {
            "file_count": len(scan_result.files),
            "skipped_file_count": len(scan_result.skipped),
            "chunk_count": len(chunks),
            "index_build_seconds": elapsed,
            "index_serialized_bytes": serialized_size,
            "index_serialized_mb": serialized_size / (1024 * 1024),
            "index_embedding_calls": tracker.calls - before_calls,
            "index_embedding_tokens": (
                tracker.input_tokens - before_tokens if tracker.usage_available else None
            ),
            "embedding_usage_available": tracker.usage_available,
        },
    )


def _search_case(
    case: dict[str, Any],
    queries: tuple[str, ...],
    *,
    chunks: tuple[SourceChunk, ...],
    index: SemanticIndex,
    tracker: UsageTracker,
    semantic_store: VectorStore | None = None,
    semantic_query_filter: dict[str, object] | None = None,
) -> tuple[dict[str, Any], list[float]]:
    evidence = _unique_evidence(case)
    if not evidence:
        return (
            {
                "case_id": case["id"],
                "repository_id": case["repository_id"],
                "applicable": False,
                "failure_class": "insufficient_evidence",
                "queries": list(queries),
                "results": [],
            },
            [],
        )

    by_key: dict[tuple[str, int, int], RankedChunk] = {}
    query_latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        bm25 = search_chunks_bm25(query, chunks, limit=SOURCE_LIMIT)
        semantic = search_chunks_semantic(
            query,
            index,
            tracker.embed,
            limit=SOURCE_LIMIT,
            min_score=SEMANTIC_MIN_SCORE,
            vector_store=semantic_store,
            query_filter=semantic_query_filter,
        )
        reranked = rerank_ranked_chunks(
            query,
            (("bm25", bm25), ("semantic", semantic)),
            limit=FINAL_LIMIT,
        )
        query_latencies.append(time.perf_counter() - started)
        for result in reranked:
            key = _key(result)
            previous = by_key.get(key)
            if previous is None or result.rank < previous.rank:
                by_key[key] = result

    results = sorted(
        by_key.values(),
        key=lambda item: (item.rank, item.chunk.relative_path, item.chunk.start_line),
    )
    expected_lines = _expected_lines(evidence)
    returned_lines: set[tuple[str, int]] = set()
    for result in results:
        returned_lines.update(_chunk_lines(result.chunk))
    covered_lines = expected_lines & returned_lines
    strict_hits = [
        any(_covers(result.chunk, item) for result in results) for item in evidence
    ]
    strict_recall = sum(strict_hits) / len(strict_hits) if strict_hits else 1.0
    complete = all(strict_hits)
    relevant_returned_lines = 0
    for result in results:
        relevant_returned_lines += len(_chunk_lines(result.chunk) & expected_lines)
    returned_line_count = sum(
        result.chunk.end_line - result.chunk.start_line + 1 for result in results
    )
    return (
        {
            "case_id": case["id"],
            "repository_id": case["repository_id"],
            "applicable": True,
            "queries": list(queries),
            "expected_evidence_count": len(evidence),
            "strict_evidence_recall": strict_recall,
            "complete_evidence_coverage": complete,
            "evidence_line_recall": (
                len(covered_lines) / len(expected_lines) if expected_lines else 1.0
            ),
            "citation_precision": (
                relevant_returned_lines / returned_line_count if returned_line_count else 0.0
            ),
            "context_lines_at_5": returned_line_count,
            "unique_context_lines_at_5": len(returned_lines),
            "duplicate_context_line_ratio": (
                1.0 - len(returned_lines) / returned_line_count
                if returned_line_count
                else 0.0
            ),
            "results": [
                {
                    "rank": result.rank,
                    "score": result.score,
                    "relative_path": result.chunk.relative_path,
                    "start_line": result.chunk.start_line,
                    "end_line": result.chunk.end_line,
                    "retrieval_reason": result.retrieval_reason,
                }
                for result in results
            ],
        },
        query_latencies,
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if row["applicable"]]
    count = len(applicable)
    if not count:
        return {"case_count": len(rows), "applicable_case_count": 0}
    failure_counts: Counter[str] = Counter()
    for row in applicable:
        if not row["complete_evidence_coverage"]:
            failure_counts["incomplete_evidence_coverage"] += 1
    return {
        "case_count": len(rows),
        "applicable_case_count": count,
        "complete_evidence_coverage_rate": sum(
            row["complete_evidence_coverage"] for row in applicable
        )
        / count,
        "evidence_recall": sum(row["strict_evidence_recall"] for row in applicable) / count,
        "mean_evidence_line_recall": sum(row["evidence_line_recall"] for row in applicable)
        / count,
        "mean_citation_precision": sum(row["citation_precision"] for row in applicable) / count,
        "mean_context_lines_at_5": sum(row["context_lines_at_5"] for row in applicable) / count,
        "mean_unique_context_lines_at_5": sum(
            row["unique_context_lines_at_5"] for row in applicable
        )
        / count,
        "mean_duplicate_context_line_ratio": sum(
            row["duplicate_context_line_ratio"] for row in applicable
        )
        / count,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _load_archived_results() -> dict[str, dict[str, Any]]:
    if not ARCHIVED_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            "缺少固定查询集归档 results.jsonl；本实验不自动重新调用 Chat 模型。"
        )
    latest: dict[str, dict[str, Any]] = {}
    for line in ARCHIVED_RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["case_id"]] = row
    return latest


def _load_repositories(
    document: dict[str, Any], selected: list[str] | None
) -> dict[str, dict[str, Any]]:
    wanted = set(selected or [item["id"] for item in document["repositories"]])
    repositories = {}
    for item in document["repositories"]:
        if item["id"] in wanted:
            repositories[item["id"]] = item
    missing = wanted - set(repositories)
    if missing:
        raise ValueError(f"未知仓库：{sorted(missing)}")
    return repositories


def run(
    *,
    output_root: Path,
    repositories: list[str] | None = None,
    profiles_to_run: Sequence[dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    archived = _load_archived_results()
    repository_metadata = _load_repositories(document, repositories)
    cases = [
        case for case in document["cases"] if case["repository_id"] in repository_metadata
    ]
    model = OpenAIEmbeddingModel.from_environment()
    started_at = datetime.now(UTC).isoformat()
    output_root.mkdir(parents=True, exist_ok=True)
    query_cache: dict[str, EmbeddingBatch] = {}
    profiles: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    selected_profiles = tuple(profiles_to_run or CHUNK_PROFILES)
    for profile in selected_profiles:
        profile_name = (
            f"lines-{profile['max_lines']}-overlap-"
            f"{int(profile['overlap_ratio'] * 100):02d}"
        )
        tracker = UsageTracker(model, query_cache=query_cache)
        by_repository: dict[str, Any] = {}
        profile_rows: list[dict[str, Any]] = []
        all_query_latencies: list[float] = []
        for repository_id, metadata in repository_metadata.items():
            root = PROJECT_ROOT / metadata["local_root"]
            chunks, index, line_counts, build_stats = _build_index(
                root,
                max_lines=profile["max_lines"],
                overlap_ratio=profile["overlap_ratio"],
                tracker=tracker,
            )
            repository_cases = [
                case for case in cases if case["repository_id"] == repository_id
            ]
            for case in repository_cases:
                queries = _queries(case, archived)
                row, latencies = _search_case(
                    case,
                    queries,
                    chunks=chunks,
                    index=index,
                    tracker=tracker,
                )
                for result in row["results"]:
                    chunk = SourceChunk(
                        result["relative_path"],
                        result["start_line"],
                        result["end_line"],
                        "",
                    )
                    result["valid"] = _valid_chunk(chunk, root, line_counts)
                if row["applicable"]:
                    expected = _unique_evidence(case)
                    row["single_chunk_coverable"] = [
                        any(_covers(chunk, item) for chunk in chunks) for item in expected
                    ]
                row["profile"] = profile_name
                profile_rows.append(row)
                all_query_latencies.extend(latencies)
            repository_rows = [
                row for row in profile_rows if row["repository_id"] == repository_id
            ]
            valid_results = [
                result
                for row in repository_rows
                for result in row["results"]
            ]
            build_stats["citation_validity"] = (
                sum(result["valid"] for result in valid_results) / len(valid_results)
                if valid_results
                else 1.0
            )
            by_repository[repository_id] = {
                "build": build_stats,
                "metrics": _aggregate(repository_rows),
            }
        profile_summary = {
            "profile": profile_name,
            "max_lines": profile["max_lines"],
            "overlap_ratio": profile["overlap_ratio"],
            "metrics": _aggregate(profile_rows),
            "by_repository": by_repository,
            "query_p95_seconds": _p95(all_query_latencies),
            "query_count": len(all_query_latencies),
            "embedding_calls_total": tracker.calls,
            "embedding_tokens_total": tracker.input_tokens if tracker.usage_available else None,
            "embedding_usage_available": tracker.usage_available,
        }
        profiles.append(profile_summary)
        all_rows.extend(profile_rows)
        (output_root / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
            encoding="utf-8",
        )
        (output_root / "summary.partial.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "EXP-001",
                    "status": "partial",
                    "completed_profiles": len(profiles),
                    "profiles": profiles,
                    "note": "中途失败时保留；不能当作九格完整结果。",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"profile": profile_name, "metrics": profile_summary["metrics"]},
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "experiment_id": "EXP-001",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "case_set": document["case_set"],
            "case_count": len(cases),
            "repositories": list(repository_metadata),
            "cases_path": str(CASES_PATH),
            "fixed_queries_path": str(ARCHIVED_RESULTS_PATH),
            "chat_calls": 0,
        },
        "fixed_pipeline": {
            "retriever": "BM25 + semantic",
            "semantic_min_score": SEMANTIC_MIN_SCORE,
            "source_limit": SOURCE_LIMIT,
            "final_limit": FINAL_LIMIT,
            "embedding_model": model.model,
            "rrf_k": 60,
            "source_weights": {"bm25": 1.0, "semantic": 0.8},
        },
        "profiles": profiles,
        "notes": [
            "This is a deterministic retrieval comparison; no Chat model call was made.",
            "Archived Router subquestions are reused so every profile sees the same queries.",
            "citation_validity is calculated against the actual frozen repository roots.",
            "index_serialized_mb is a comparable local Python serialization size, not RSS.",
            "A final chunking decision requires manual review of metrics and failure cases; "
            "the runner does not invent a threshold.",
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repository", action="append")
    parser.add_argument(
        "--profile",
        action="append",
        metavar="LINES:OVERLAP",
        help="只运行指定 profile，例如 120:0.1；可重复传入。",
    )
    args = parser.parse_args()
    selected_profiles = None
    if args.profile:
        selected_profiles = []
        available = {
            f"{item['max_lines']}:{item['overlap_ratio']}": item for item in CHUNK_PROFILES
        }
        for value in args.profile:
            try:
                lines_text, overlap_text = value.split(":", 1)
                key = f"{int(lines_text)}:{float(overlap_text)}"
            except ValueError as error:
                raise SystemExit(f"非法 profile：{value}，格式应为 LINES:OVERLAP") from error
            profile = available.get(key)
            if profile is None:
                raise SystemExit(f"不支持的 profile：{value}，可选值为 {sorted(available)}")
            selected_profiles.append(profile)
    if not args.confirm_run:
        document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        selected = _load_repositories(document, args.repository)
        case_count = sum(
            case["repository_id"] in selected for case in document["cases"]
        )
        print(
            json.dumps(
                {
                    "experiment_id": "EXP-001",
                    "profiles": len(selected_profiles or CHUNK_PROFILES),
                    "case_count": case_count,
                    "repositories": list(selected),
                    "chat_calls": 0,
                    "embedding_calls": "real provider calls on --confirm-run",
                },
                ensure_ascii=False,
            )
        )
        return 0
    try:
        payload = run(
            output_root=args.output_root,
            repositories=args.repository,
            profiles_to_run=selected_profiles,
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "experiment_id": "EXP-001",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "note": "保留已完成 profile 的 summary.partial.json；未完成 profile 不计入实验结论。",
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"output_root": str(args.output_root), "profiles": payload["profiles"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
