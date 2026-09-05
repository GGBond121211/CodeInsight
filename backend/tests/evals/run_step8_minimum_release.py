"""Run the bounded Step 8 baseline pilot with explicit token accounting."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codeinsight.application.search_repository import HYBRID_CANDIDATE_LIMIT
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.reranker import OpenAITextReranker
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.bm25 import search_chunks_bm25
from codeinsight.retrieval.hybrid import fuse_ranked_chunks, rerank_ranked_chunks
from codeinsight.retrieval.persistent_semantic import build_persistent_semantic_index
from codeinsight.retrieval.semantic import search_chunks_semantic

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
MASTER_PATH = BACKEND_ROOT / "tests" / "evals" / "master_200_cases.json"
MANIFEST_PATH = BACKEND_ROOT / "tests" / "evals" / "step8_minimum_release_cases.json"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "STEP8-MINIMUM-RELEASE"
HARD_STOP_TOKENS = 9_000_000

REPOSITORIES = {
    "sample_repo": BACKEND_ROOT / "tests" / "fixtures" / "sample_repo",
    "httpx": PROJECT_ROOT / "work" / "benchmarks" / "httpx",
    "click": PROJECT_ROOT / "work" / "benchmarks" / "click",
    "requests": PROJECT_ROOT / "work" / "benchmarks" / "requests",
}

FUSION_PROFILES = (
    {"id": "rrf-k-20", "rrf_k": 20, "semantic_weight": 0.8, "min_score": 0.20},
    {"id": "baseline", "rrf_k": 60, "semantic_weight": 0.8, "min_score": 0.20},
    {"id": "rrf-k-100", "rrf_k": 100, "semantic_weight": 0.8, "min_score": 0.20},
    {"id": "semantic-weight-0.5", "rrf_k": 60, "semantic_weight": 0.5, "min_score": 0.20},
    {"id": "semantic-weight-1.0", "rrf_k": 60, "semantic_weight": 1.0, "min_score": 0.20},
    {"id": "min-score-0.0", "rrf_k": 60, "semantic_weight": 0.8, "min_score": 0.0},
    {"id": "min-score-0.35", "rrf_k": 60, "semantic_weight": 0.8, "min_score": 0.35},
)


class UsageEmbedding:
    """Cache identical embedding inputs and expose conservative usage totals."""

    def __init__(self, model: OpenAIEmbeddingModel) -> None:
        self.model = model.model
        self._model = model
        self._cache: dict[tuple[str, ...], EmbeddingBatch] = {}
        self.calls = 0
        self.actual_tokens = 0
        self.estimated_tokens = 0

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        key = tuple(texts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        batch = self._model.embed(key)
        actual = getattr(batch, "input_tokens", None)
        estimate = max(1, sum(len(text.encode("utf-8")) for text in key) // 4)
        self.calls += 1
        self.actual_tokens += int(actual or 0)
        self.estimated_tokens += int(actual or estimate)
        self._cache[key] = batch
        return batch


def _expected_evidence(case: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    expected = case.get("expected", {})
    values = list(expected.get("evidence", ()))
    if not values:
        for subquestion in expected.get("subquestions", ()):
            values.extend(subquestion.get("evidence", ()))
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in values:
        key = (item["path"], item["start_line"], item["end_line"])
        unique[key] = item
    return tuple(unique.values())


def _covers(result: RankedChunk, expected: dict[str, Any]) -> bool:
    chunk = result.chunk
    return (
        chunk.relative_path == expected["path"]
        and chunk.start_line <= expected["start_line"]
        and chunk.end_line >= expected["end_line"]
    )


def _recall(results: Sequence[RankedChunk], expected: Sequence[dict[str, Any]]) -> bool | None:
    if not expected:
        return None
    return all(any(_covers(result, item) for result in results) for item in expected)


def _load_cases() -> list[dict[str, Any]]:
    master = json.loads(MASTER_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in master["cases"]}
    return [by_id[case_id] for case_id in manifest["case_ids"]]


def _build_indexes(
    cases: Sequence[dict[str, Any]],
    embed: UsageEmbedding,
) -> dict[str, SemanticIndex]:
    indexes: dict[str, SemanticIndex] = {}
    for repository_id in sorted({case["repository_id"] for case in cases}):
        root = REPOSITORIES[repository_id]
        if not root.is_dir():
            raise FileNotFoundError(f"缺少冻结仓库：{root}")
        indexes[repository_id] = build_persistent_semantic_index(
            root,
            scan_repository(root),
            embed.embed,
            model=embed.model,
            chunk_max_lines=80,
        )
        print(
            json.dumps(
                {
                    "event": "index_ready",
                    "repository_id": repository_id,
                    "entries": len(indexes[repository_id].entries),
                },
                ensure_ascii=False,
            )
        )
    return indexes


def _fusion_row(
    case: dict[str, Any],
    profile: dict[str, Any],
    *,
    bm25: Sequence[RankedChunk],
    semantic: Sequence[RankedChunk],
) -> dict[str, Any]:
    fused = fuse_ranked_chunks(
        (("bm25", bm25), ("semantic", semantic)),
        limit=HYBRID_CANDIDATE_LIMIT,
        rrf_k=profile["rrf_k"],
        source_weights={"bm25": 1.0, "semantic": profile["semantic_weight"]},
    )
    expected = _expected_evidence(case)
    return {
        "case_id": case["id"],
        "repository_id": case["repository_id"],
        "category": case["category"],
        "profile": profile["id"],
        "rrf_k": profile["rrf_k"],
        "semantic_weight": profile["semantic_weight"],
        "semantic_min_score": profile["min_score"],
        "candidate_count": len(fused),
        "candidate_recall_at_10": _recall(fused[:10], expected),
        "candidate_recall_at_20": _recall(fused[:20], expected),
        "candidate_recall_at_100": _recall(fused, expected),
        "expected_evidence_count": len(expected),
    }


def run() -> dict[str, Any]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-step8-minimum")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    embedding = UsageEmbedding(OpenAIEmbeddingModel.from_environment())
    reranker = OpenAITextReranker.from_environment()
    indexes = _build_indexes(cases, embedding)
    rows_path = output_dir / "rows.jsonl"
    fusion_path = output_dir / "fusion_scan.jsonl"
    rerank_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    charged_tokens = embedding.estimated_tokens
    rerank_charged_tokens = 0

    with rows_path.open("w", encoding="utf-8") as rows_file, fusion_path.open(
        "w", encoding="utf-8"
    ) as fusion_file:
        for case in cases:
            if charged_tokens >= HARD_STOP_TOKENS:
                raise RuntimeError("Step 8 pilot 达到 9,000,000 Token 硬停止线")
            question = case["input"]["question"]
            bm25 = search_chunks_bm25(
                question,
                tuple(
                    item.chunk
                    for item in indexes[case["repository_id"]].entries
                ),
                limit=HYBRID_CANDIDATE_LIMIT,
            )
            semantic_by_score: dict[float, tuple[RankedChunk, ...]] = {}
            for profile in FUSION_PROFILES:
                min_score = profile["min_score"]
                if min_score not in semantic_by_score:
                    semantic_by_score[min_score] = search_chunks_semantic(
                        question,
                        indexes[case["repository_id"]],
                        embedding.embed,
                        limit=HYBRID_CANDIDATE_LIMIT,
                        min_score=min_score,
                    )
                    charged_tokens = embedding.estimated_tokens + rerank_charged_tokens
                row = _fusion_row(
                    case,
                    profile,
                    bm25=bm25,
                    semantic=semantic_by_score[min_score],
                )
                fusion_rows.append(row)
                fusion_file.write(json.dumps(row, ensure_ascii=False) + "\n")

            result = rerank_ranked_chunks(
                question,
                (
                    ("bm25", bm25),
                    ("semantic", semantic_by_score[0.20]),
                ),
                reranker=reranker,
                limit=10,
                candidate_limit=HYBRID_CANDIDATE_LIMIT,
                rrf_k=60,
                source_weights={"bm25": 1.0, "semantic": 0.8},
            )
            telemetry = reranker.last_call
            if telemetry is None:
                raise RuntimeError("Rerank 调用没有产生 telemetry")
            rerank_charged_tokens += telemetry.total_tokens or telemetry.estimated_input_tokens
            charged_tokens = embedding.estimated_tokens + rerank_charged_tokens
            if charged_tokens >= HARD_STOP_TOKENS:
                raise RuntimeError("Step 8 pilot 达到 9,000,000 Token 硬停止线")
            row = {
                "case_id": case["id"],
                "repository_id": case["repository_id"],
                "category": case["category"],
                "expected_evidence_count": len(_expected_evidence(case)),
                "rerank_count": len(result),
                "rerank_recall_at_10": _recall(result, _expected_evidence(case)),
                "telemetry": telemetry.__dict__,
                "charged_tokens": telemetry.total_tokens or telemetry.estimated_input_tokens,
            }
            rerank_rows.append(row)
            rows_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_file.flush()
            fusion_file.flush()
            print(json.dumps({"event": "case_done", **row}, ensure_ascii=False))

    summary = {
        "schema_version": 1,
        "experiment_id": "STEP8-MINIMUM-RELEASE",
        "run_id": run_id,
        "status": "completed",
        "case_count": len(cases),
        "fusion_profile_count": len(FUSION_PROFILES),
        "rerank_call_count": len(rerank_rows),
        "embedding": {
            "model": embedding.model,
            "calls": embedding.calls,
            "actual_tokens": embedding.actual_tokens,
            "estimated_tokens": embedding.estimated_tokens,
        },
        "rerank": {
            "model": reranker.model,
            "actual_tokens": sum(row["telemetry"]["total_tokens"] or 0 for row in rerank_rows),
            "estimated_tokens": sum(row["charged_tokens"] for row in rerank_rows),
            "latency_ms": [row["telemetry"]["latency_ms"] for row in rerank_rows],
        },
        "charged_tokens_upper_bound": charged_tokens,
        "budget_hard_stop": HARD_STOP_TOKENS,
        "outputs": {
            "rows": str(rows_path),
            "fusion_scan": str(fusion_path),
        },
        "notes": [
            "Fusion scan is deterministic candidate-stage evidence, not final answer quality.",
            "Rerank quality is a 30-case pilot and is not a full Master-200 conclusion.",
            "Fake Provider concurrency is evaluated separately and is not represented here.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
