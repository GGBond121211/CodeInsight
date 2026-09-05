"""Compare the fixed-line baseline with structural code chunking.

This runner has a dry-run mode by default. ``--confirm-run`` is required before
real Embedding calls are made because structural chunking creates substantially
more chunks than the fixed-line baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_exp001_chunk_overlap import (
    UsageTracker,
    _aggregate,
    _load_archived_results,
    _load_repositories,
    _queries,
    _search_case,
)

from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.ingestion.structural_chunker import chunk_structural_scan_result
from codeinsight.retrieval.semantic import build_semantic_index

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = Path(__file__).with_name("master_200_cases.json")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "EXP-001B"


def _profile_chunks(root: Path, profile: str):
    scan_result = scan_repository(root)
    if profile == "fixed-80-00":
        chunks = chunk_scan_result(scan_result, max_lines=80, overlap_ratio=0.0)
    elif profile == "structural-v1":
        chunks = chunk_structural_scan_result(
            scan_result,
            fallback_max_lines=80,
            max_structure_lines=120,
        )
    else:
        raise ValueError(f"不支持的 profile：{profile}")
    source_bytes = 0
    for chunk in chunks:
        source_bytes += len(chunk.text.encode("utf-8"))
    return scan_result, chunks, source_bytes


def _dry_run(document: dict[str, Any], repositories: list[str] | None) -> dict[str, Any]:
    metadata = _load_repositories(document, repositories)
    counts = {}
    for profile in ("fixed-80-00", "structural-v1"):
        profile_stats = {}
        total_chunks = 0
        total_bytes = 0
        for repo_id, item in metadata.items():
            scan_result, chunks, source_bytes = _profile_chunks(
                PROJECT_ROOT / item["local_root"], profile
            )
            profile_stats[repo_id] = {
                "files": len(scan_result.files),
                "chunks": len(chunks),
                "utf8_bytes": source_bytes,
            }
            total_chunks += len(chunks)
            total_bytes += source_bytes
        counts[profile] = {
            "repositories": profile_stats,
            "total_chunks": total_chunks,
            "total_utf8_bytes": total_bytes,
        }
    return counts


def run(*, output_root: Path, repositories: list[str] | None = None) -> dict[str, Any]:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    archived = _load_archived_results()
    metadata = _load_repositories(document, repositories)
    cases = [case for case in document["cases"] if case["repository_id"] in metadata]
    model = OpenAIEmbeddingModel.from_environment()
    query_cache = {}
    profiles = []
    all_rows = []
    for profile in ("fixed-80-00", "structural-v1"):
        tracker = UsageTracker(model, query_cache=query_cache)
        rows = []
        build_stats_by_repo = {}
        query_latencies = []
        for repo_id, item in metadata.items():
            root = PROJECT_ROOT / item["local_root"]
            scan_result, chunks, source_bytes = _profile_chunks(root, profile)
            before_calls = tracker.calls
            before_tokens = tracker.input_tokens
            started = time.perf_counter()
            index = build_semantic_index(chunks, tracker.embed)
            build_seconds = time.perf_counter() - started
            build_stats_by_repo[repo_id] = {
                "file_count": len(scan_result.files),
                "chunk_count": len(chunks),
                "source_utf8_bytes": source_bytes,
                "index_serialized_bytes": len(pickle.dumps(index, protocol=5)),
                "index_build_seconds": build_seconds,
                "index_embedding_calls": tracker.calls - before_calls,
                "index_embedding_tokens": (
                    tracker.input_tokens - before_tokens if tracker.usage_available else None
                ),
            }
            for case in cases:
                if case["repository_id"] != repo_id:
                    continue
                row, latencies = _search_case(
                    case,
                    _queries(case, archived),
                    chunks=chunks,
                    index=index,
                    tracker=tracker,
                )
                row["profile"] = profile
                rows.append(row)
                query_latencies.extend(latencies)
        profile_payload = {
            "profile": profile,
            "metrics": _aggregate(rows),
            "build_by_repository": build_stats_by_repo,
            "query_count": len(query_latencies),
            "query_p95_seconds": _p95(query_latencies),
            "embedding_calls_total": tracker.calls,
            "embedding_tokens_total": tracker.input_tokens if tracker.usage_available else None,
        }
        profiles.append(profile_payload)
        all_rows.extend(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "EXP-001B",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "case_set": document["case_set"],
            "case_count": len(cases),
            "repositories": list(metadata),
            "chat_calls": 0,
        },
        "profiles": profiles,
        "notes": [
            "fixed-80-00 is the EXP-001 baseline.",
            "structural-v1 uses AST top-level class/function/module regions and "
            "fixed-line fallback.",
            "No Chat model call is made; Embedding calls require --confirm-run.",
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repository", action="append")
    args = parser.parse_args()
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not args.confirm_run:
        counts = _dry_run(document, args.repository)
        print(
            json.dumps(
                {
                    "experiment_id": "EXP-001B",
                    "profiles": counts,
                    "chat_calls": 0,
                    "embedding_calls": "real provider calls on --confirm-run",
                },
                ensure_ascii=False,
            )
        )
        return 0
    payload = run(output_root=args.output_root, repositories=args.repository)
    print(
        json.dumps(
            {"output_root": str(args.output_root), "profiles": payload["profiles"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
