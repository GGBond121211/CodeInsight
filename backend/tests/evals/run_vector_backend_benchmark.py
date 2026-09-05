"""Compare local exact cosine with Qdrant/HNSW on the frozen query set.

The runner makes real Embedding calls only with ``--confirm-run`` and never
calls a Chat model. The same fixed chunks and query Embeddings are sent to
both backends, so the comparison isolates backend behavior.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from run_exp001_chunk_overlap import (
    UsageTracker,
    _aggregate,
    _load_archived_results,
    _load_repositories,
    _p95,
    _queries,
    _search_case,
)

from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.index_pipeline import (
    publish_semantic_index,
    source_fingerprint,
    vector_points_from_semantic_index,
)
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.semantic import build_semantic_index
from codeinsight.retrieval.vector_store import LocalJsonVectorStore, VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = Path(__file__).with_name("master_200_cases.json")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "EXP-006"


def _mrr(rows: list[dict[str, Any]]) -> float | None:
    applicable = [row for row in rows if row["applicable"]]
    if not applicable:
        return None
    values: list[float] = []
    for row in applicable:
        expected = row.get("expected_evidence", [])
        reciprocal = 0.0
        for result in row["results"]:
            if any(
                result["relative_path"] == item["path"]
                and result["start_line"] <= item["start_line"]
                and result["end_line"] >= item["end_line"]
                for item in expected
            ):
                reciprocal = 1.0 / result["rank"]
                break
        values.append(reciprocal)
    return sum(values) / len(values)


def _citation_validity(rows: list[dict[str, Any]], line_counts: dict[str, int]) -> float:
    total = 0
    valid = 0
    for row in rows:
        for result in row["results"]:
            total += 1
            if (
                result["relative_path"] in line_counts
                and result["start_line"] >= 1
                and result["end_line"] >= result["start_line"]
                and result["end_line"] <= line_counts[result["relative_path"]]
            ):
                valid += 1
    return valid / total if total else 1.0


def _expected_evidence(case: dict[str, Any]) -> list[dict[str, Any]]:
    expected = case.get("expected", {})
    values = list(expected.get("evidence", ()))
    for subquestion in expected.get("subquestions", ()):
        values.extend(subquestion.get("evidence", ()))
    return values


def _index_for_repo(root: Path, tracker: UsageTracker):
    scan_result = scan_repository(root)
    chunks = chunk_scan_result(scan_result, max_lines=80, overlap_ratio=0.0)
    started = time.perf_counter()
    index = build_semantic_index(chunks, tracker.embed)
    return scan_result, chunks, index, time.perf_counter() - started


def _store_for_backend(
    backend: str,
    *,
    output_root: Path,
    repo_id: str,
    index,
    scan_result,
    qdrant_url: str,
) -> tuple[VectorStore, dict[str, Any]]:
    fingerprints = {
        source.relative_path: source_fingerprint(source.text)
        for source in scan_result.files
    }
    points = vector_points_from_semantic_index(
        index,
        repo_id=repo_id,
        source_fingerprints=fingerprints,
        chunk_version="fixed-lines-v1",
    )
    if backend == "local_json":
        store = LocalJsonVectorStore(output_root / "local" / f"{repo_id}.json")
        started = time.perf_counter()
        publish_semantic_index(
            index,
            store,
            repo_id=repo_id,
            source_fingerprints=fingerprints,
            chunk_version="fixed-lines-v1",
        )
        return store, {
            "backend": "local_json",
            "collection": str(store.path),
            "publish_seconds": time.perf_counter() - started,
            "point_count": len(points),
            "fallback_reason": None,
        }

    if backend != "qdrant_hnsw":
        raise ValueError(f"不支持的向量后端：{backend}")
    client = QdrantClient(url=qdrant_url, timeout=5)
    collection = f"codeinsight_exp002_{repo_id}_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        dimensions=index.metadata.dimensions,
        hnsw_m=16,
        hnsw_ef_construction=128,
        hnsw_ef_search=64,
    )
    started = time.perf_counter()
    publish_semantic_index(
        index,
        store,
        repo_id=repo_id,
        source_fingerprints=fingerprints,
        chunk_version="fixed-lines-v1",
    )
    return store, {
        "backend": "qdrant_hnsw",
        "collection": collection,
        "publish_seconds": time.perf_counter() - started,
        "point_count": len(points),
        "fallback_reason": None,
    }


def run(*, output_root: Path, qdrant_url: str) -> dict[str, Any]:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    archived = _load_archived_results()
    metadata = _load_repositories(document, None)
    cases = [case for case in document["cases"] if case["repository_id"] in metadata]
    model = OpenAIEmbeddingModel.from_environment()
    query_cache: dict[str, EmbeddingBatch] = {}
    profiles: list[dict[str, Any]] = []

    for backend in ("local_json", "qdrant_hnsw"):
        tracker = UsageTracker(model, query_cache=query_cache)
        rows: list[dict[str, Any]] = []
        query_latencies: list[float] = []
        repository_stats: dict[str, Any] = {}
        for repo_id, item in metadata.items():
            root = PROJECT_ROOT / item["local_root"]
            scan_result, chunks, index, build_seconds = _index_for_repo(root, tracker)
            line_counts = {
                source.relative_path: len(
                    source.text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
                )
                for source in scan_result.files
            }
            store, store_stats = _store_for_backend(
                backend,
                output_root=output_root,
                repo_id=repo_id,
                index=index,
                scan_result=scan_result,
                qdrant_url=qdrant_url,
            )
            before_calls = tracker.calls
            before_tokens = tracker.input_tokens
            repo_rows: list[dict[str, Any]] = []
            for case in cases:
                if case["repository_id"] != repo_id:
                    continue
                row, latencies = _search_case(
                    case,
                    _queries(case, archived),
                    chunks=chunks,
                    index=index,
                    tracker=tracker,
                    semantic_store=store,
                    semantic_query_filter={"repoId": repo_id, "visibility": "active"},
                )
                expected = _expected_evidence(case)
                row["expected_evidence"] = expected
                rows.append(row)
                repo_rows.append(row)
                query_latencies.extend(latencies)
            repository_stats[repo_id] = {
                "file_count": len(scan_result.files),
                "chunk_count": len(index.entries),
                "index_build_seconds": build_seconds,
                "index_embedding_calls": tracker.calls - before_calls,
                "index_embedding_tokens": (
                    tracker.input_tokens - before_tokens
                    if tracker.usage_available
                    else None
                ),
                "index_backend": store_stats,
                "citation_validity": _citation_validity(repo_rows, line_counts),
            }
        metrics = _aggregate(rows)
        metrics["mrr"] = _mrr(rows)
        metrics["citation_validity"] = sum(
            item["citation_validity"] for item in repository_stats.values()
        ) / len(repository_stats)
        profiles.append(
            {
                "backend": backend,
                "hnsw": (
                    {"m": 16, "ef_construction": 128, "ef_search": 64}
                    if backend == "qdrant_hnsw"
                    else None
                ),
                "metrics": metrics,
                "repositories": repository_stats,
                "query_count": len(query_latencies),
                "query_p95_seconds": _p95(query_latencies),
                "embedding_calls_total": tracker.calls,
                "embedding_tokens_total": (
                    tracker.input_tokens if tracker.usage_available else None
                ),
                "trace": {
                    "backend": backend,
                    "fallback_reason": None,
                },
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_id": "EXP-006",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "case_set": document["case_set"],
            "case_count": len(cases),
            "repositories": list(metadata),
            "chunk_profile": "fixed-80-00",
            "chat_calls": 0,
        },
        "profiles": profiles,
        "notes": [
            "Both backends use the same fixed chunks and cached query Embeddings.",
            "Evidence is reconstructed from the SemanticIndex, never from untrusted payload text.",
            "Qdrant parameters are one measured baseline, not a tuned optimum.",
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6335")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "experiment_id": "EXP-006",
                    "profiles": ["local_json", "qdrant_hnsw"],
                    "chunk_profile": "fixed-80-00",
                    "chat_calls": 0,
                    "embedding_calls": "real provider calls on --confirm-run",
                },
                ensure_ascii=False,
            )
        )
        return 0
    payload = run(output_root=args.output_root, qdrant_url=args.qdrant_url)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "profiles": [
                    {
                        "backend": item["backend"],
                        "metrics": item["metrics"],
                        "query_p95_seconds": item["query_p95_seconds"],
                    }
                    for item in payload["profiles"]
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
