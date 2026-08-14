"""Diagnose retrieval-stage evidence loss for Click and Requests Master-200 cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from codeinsight.application.search_repository import build_repository_semantic_index
from codeinsight.evaluation.answer_metrics import citation_covers
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.bm25 import search_chunks_bm25
from codeinsight.retrieval.hybrid import fuse_ranked_chunks, rerank_ranked_chunks
from codeinsight.retrieval.semantic import search_chunks_semantic

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
CASES_PATH = Path(__file__).with_name("master_200_cases.json")
RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evals"
    / "master-200"
    / "20260813T-master200-final-v2"
    / "results.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evals"
    / "master-200"
    / "20260813T-master200-final-v2"
    / "real-repositories-retrieval-funnel-v2.json"
)
SOURCE_LIMIT = 20
FINAL_LIMIT = 5
CHUNK_MAX_LINES = 80


def _key(result) -> tuple[str, int, int]:
    return result.chunk.relative_path, result.chunk.start_line, result.chunk.end_line


def _chunk(result):
    return getattr(result, "chunk", result)


def _covers(result, expected: dict[str, Any]) -> bool:
    chunk = _chunk(result)
    return (
        chunk.relative_path == expected["path"]
        and chunk.start_line <= expected["start_line"]
        and chunk.end_line >= expected["end_line"]
    )


def _overlaps(result, expected: dict[str, Any]) -> bool:
    chunk = _chunk(result)
    return (
        chunk.relative_path == expected["path"]
        and chunk.start_line <= expected["end_line"]
        and chunk.end_line >= expected["start_line"]
    )


def _citation_overlaps(citation: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        citation["relative_path"] == expected["path"]
        and citation["start_line"] <= expected["end_line"]
        and citation["end_line"] >= expected["start_line"]
    )


def _latest_results() -> dict[str, dict[str, Any]]:
    latest = {}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["case_id"]] = row
    return latest


def _expected(case: dict[str, Any]) -> list[dict[str, Any]]:
    values = list(case["expected"].get("evidence", ()))
    for subquestion in case["expected"].get("subquestions", ()):
        values.extend(subquestion.get("evidence", ()))
    unique = {}
    for item in values:
        unique[(item["path"], item["start_line"], item["end_line"])] = item
    return list(unique.values())


def _stage_metrics(expected: list[dict[str, Any]], results) -> dict[str, Any]:
    strict_hits = [any(_covers(result, item) for result in results) for item in expected]
    overlap_hits = [any(_overlaps(result, item) for result in results) for item in expected]
    return {
        "strict_recall": sum(strict_hits) / len(strict_hits) if strict_hits else 1.0,
        "overlap_recall": sum(overlap_hits) / len(overlap_hits) if overlap_hits else 1.0,
        "strict_complete": all(strict_hits),
        "overlap_complete": all(overlap_hits),
    }


def _citation_metrics(expected: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    citations = (row.get("result") or {}).get("citations", ())
    strict_hits = [
        any(
            citation_covers(
                type(
                    "Citation",
                    (),
                    {
                        "relative_path": citation["relative_path"],
                        "start_line": citation["start_line"],
                        "end_line": citation["end_line"],
                    },
                )(),
                item,
            )
            for citation in citations
        )
        for item in expected
    ]
    overlap_hits = [
        any(_citation_overlaps(citation, item) for citation in citations) for item in expected
    ]
    return {
        "strict_recall": sum(strict_hits) / len(strict_hits) if strict_hits else 1.0,
        "overlap_recall": sum(overlap_hits) / len(overlap_hits) if overlap_hits else 1.0,
        "strict_complete": all(strict_hits),
        "overlap_complete": all(overlap_hits),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ("raw_union", "fused_top20", "reranked_top5", "final_citations")
    result = {
        "case_count": len(rows),
        "expected_evidence_count": sum(row["expected_evidence_count"] for row in rows),
        "annotation_single_chunk_coverable_rate": mean(
            row["annotation_single_chunk_coverable_rate"] for row in rows
        ),
    }
    for stage in stages:
        result[stage] = {
            "strict_recall": mean(row[stage]["strict_recall"] for row in rows),
            "overlap_recall": mean(row[stage]["overlap_recall"] for row in rows),
            "strict_complete_case_rate": mean(row[stage]["strict_complete"] for row in rows),
            "overlap_complete_case_rate": mean(row[stage]["overlap_complete"] for row in rows),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", action="append", choices=("httpx", "click", "requests"))
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    repositories = args.repository or ["httpx", "click", "requests"]
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    metadata = {item["id"]: item for item in document["repositories"]}
    cases = [case for case in document["cases"] if case["repository_id"] in repositories]
    latest = _latest_results()
    planned_query_count = sum(
        max(1, len(latest[case["id"]]["router"]["plan"]["subquestions"])) for case in cases
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "repositories": dict(Counter(case["repository_id"] for case in cases)),
                "embedding_index_builds": len(repositories),
                "embedding_query_calls": planned_query_count,
                "chat_calls": 0,
            },
            ensure_ascii=False,
        )
    )
    if not args.confirm_run:
        return 0

    embed_model = OpenAIEmbeddingModel.from_environment()
    rows = []
    for repository_id in repositories:
        root = PROJECT_ROOT / metadata[repository_id]["local_root"]
        chunks = chunk_scan_result(scan_repository(root), max_lines=CHUNK_MAX_LINES)
        semantic_index = build_repository_semantic_index(
            root, chunk_max_lines=CHUNK_MAX_LINES, semantic_embed=embed_model.embed
        )
        for case in (item for item in cases if item["repository_id"] == repository_id):
            row = latest[case["id"]]
            plan = row["router"]["plan"]
            queries = [item["question"] for item in plan["subquestions"]] or [
                case["input"]["question"]
            ]
            raw_by_key = {}
            fused_by_key = {}
            reranked_by_key = {}
            for query in queries:
                bm25 = search_chunks_bm25(query, chunks, limit=SOURCE_LIMIT)
                semantic = search_chunks_semantic(
                    query, semantic_index, embed_model.embed, limit=SOURCE_LIMIT
                )
                sources = (("bm25", bm25), ("semantic", semantic))
                for item in (value for _, values in sources for value in values):
                    raw_by_key[_key(item)] = item
                for item in fuse_ranked_chunks(sources, limit=SOURCE_LIMIT):
                    fused_by_key[_key(item)] = item
                for item in rerank_ranked_chunks(query, sources, limit=FINAL_LIMIT):
                    reranked_by_key[_key(item)] = item
            raw = tuple(raw_by_key.values())
            fused = tuple(fused_by_key.values())
            reranked = tuple(reranked_by_key.values())
            expected = _expected(case)
            coverable = [any(_covers(chunk, item) for chunk in chunks) for item in expected]
            rows.append(
                {
                    "case_id": case["id"],
                    "repository_id": repository_id,
                    "queries": queries,
                    "expected_evidence_count": len(expected),
                    "annotation_single_chunk_coverable_rate": (
                        sum(coverable) / len(coverable) if coverable else 1.0
                    ),
                    "raw_union": _stage_metrics(expected, raw),
                    "fused_top20": _stage_metrics(expected, fused),
                    "reranked_top5": _stage_metrics(expected, reranked),
                    "final_citations": _citation_metrics(expected, row),
                }
            )
            print(
                json.dumps({"case_id": case["id"], "done": len(rows)}, ensure_ascii=False),
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "case_count": len(rows),
        "summary": _summarize(rows),
        "by_repository": {
            repository_id: _summarize(
                [row for row in rows if row["repository_id"] == repository_id]
            )
            for repository_id in repositories
        },
        "rows": rows,
        "notes": [
            "Strict recall requires one chunk or citation to contain the full expected span.",
            "Overlap recall requires the same file and any line overlap with expected evidence.",
            "The diagnostic reuses each final Router query and does not call the chat model.",
            "Each repository semantic index is built once and reused for all its cases.",
        ],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"output": str(OUTPUT_PATH), "summary": payload["summary"]}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
