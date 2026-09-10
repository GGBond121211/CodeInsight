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
from codeinsight.infrastructure.reranker import OpenAITextReranker
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.hybrid import fuse_ranked_chunks, rerank_ranked_chunks
from codeinsight.retrieval.semantic import search_chunks_dense
from codeinsight.retrieval.sparse import search_chunks_sparse

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
    strict_hits = []
    for item in expected:
        item_is_covered = False
        for result in results:
            if _covers(result, item):
                item_is_covered = True
                break
        strict_hits.append(item_is_covered)

    overlap_hits = []
    for item in expected:
        item_is_overlapped = False
        for result in results:
            if _overlaps(result, item):
                item_is_overlapped = True
                break
        overlap_hits.append(item_is_overlapped)

    strict_hit_count = 0
    for was_hit in strict_hits:
        if was_hit:
            strict_hit_count += 1
    overlap_hit_count = 0
    for was_hit in overlap_hits:
        if was_hit:
            overlap_hit_count += 1
    strict_complete = True
    for was_hit in strict_hits:
        if not was_hit:
            strict_complete = False
            break
    overlap_complete = True
    for was_hit in overlap_hits:
        if not was_hit:
            overlap_complete = False
            break
    return {
        "strict_recall": strict_hit_count / len(strict_hits) if strict_hits else 1.0,
        "overlap_recall": overlap_hit_count / len(overlap_hits) if overlap_hits else 1.0,
        "strict_complete": strict_complete,
        "overlap_complete": overlap_complete,
    }


def _citation_metrics(expected: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    citations = (row.get("result") or {}).get("citations", ())
    strict_hits = []
    for item in expected:
        item_is_covered = False
        for citation in citations:
            citation_object = type(
                "Citation",
                (),
                {
                    "relative_path": citation["relative_path"],
                    "start_line": citation["start_line"],
                    "end_line": citation["end_line"],
                },
            )()
            if citation_covers(citation_object, item):
                item_is_covered = True
                break
        strict_hits.append(item_is_covered)

    overlap_hits = []
    for item in expected:
        item_is_overlapped = False
        for citation in citations:
            if _citation_overlaps(citation, item):
                item_is_overlapped = True
                break
        overlap_hits.append(item_is_overlapped)

    strict_hit_count = 0
    for was_hit in strict_hits:
        if was_hit:
            strict_hit_count += 1
    overlap_hit_count = 0
    for was_hit in overlap_hits:
        if was_hit:
            overlap_hit_count += 1
    strict_complete = True
    for was_hit in strict_hits:
        if not was_hit:
            strict_complete = False
            break
    overlap_complete = True
    for was_hit in overlap_hits:
        if not was_hit:
            overlap_complete = False
            break
    return {
        "strict_recall": strict_hit_count / len(strict_hits) if strict_hits else 1.0,
        "overlap_recall": overlap_hit_count / len(overlap_hits) if overlap_hits else 1.0,
        "strict_complete": strict_complete,
        "overlap_complete": overlap_complete,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ("raw_union", "fused_top20", "reranked_top5", "final_citations")
    expected_evidence_counts = []
    coverable_rates = []
    for row in rows:
        expected_evidence_counts.append(row["expected_evidence_count"])
        coverable_rates.append(row["annotation_single_chunk_coverable_rate"])
    result = {
        "case_count": len(rows),
        "expected_evidence_count": sum(expected_evidence_counts),
        "annotation_single_chunk_coverable_rate": mean(coverable_rates),
    }
    for stage in stages:
        strict_recall_values = []
        overlap_recall_values = []
        strict_complete_values = []
        overlap_complete_values = []
        for row in rows:
            strict_recall_values.append(row[stage]["strict_recall"])
            overlap_recall_values.append(row[stage]["overlap_recall"])
            strict_complete_values.append(row[stage]["strict_complete"])
            overlap_complete_values.append(row[stage]["overlap_complete"])
        result[stage] = {
            "strict_recall": mean(strict_recall_values),
            "overlap_recall": mean(overlap_recall_values),
            "strict_complete_case_rate": mean(strict_complete_values),
            "overlap_complete_case_rate": mean(overlap_complete_values),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", action="append", choices=("httpx", "click", "requests"))
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    repositories = args.repository or ["httpx", "click", "requests"]
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    metadata = {}
    for item in document["repositories"]:
        metadata[item["id"]] = item
    cases = []
    for case in document["cases"]:
        if case["repository_id"] in repositories:
            cases.append(case)
    latest = _latest_results()
    planned_query_count = 0
    for case in cases:
        subquestion_count = len(latest[case["id"]]["router"]["plan"]["subquestions"])
        planned_query_count += max(1, subquestion_count)
    repository_case_ids = []
    for case in cases:
        repository_case_ids.append(case["repository_id"])
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "repositories": dict(Counter(repository_case_ids)),
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
    reranker = OpenAITextReranker.from_environment()
    rows = []
    for repository_id in repositories:
        root = PROJECT_ROOT / metadata[repository_id]["local_root"]
        chunks = chunk_scan_result(scan_repository(root), max_lines=CHUNK_MAX_LINES)
        semantic_index = build_repository_semantic_index(
            root, chunk_max_lines=CHUNK_MAX_LINES, semantic_embed=embed_model.embed
        )
        repository_cases = []
        for item in cases:
            if item["repository_id"] == repository_id:
                repository_cases.append(item)
        for case in repository_cases:
            row = latest[case["id"]]
            plan = row["router"]["plan"]
            queries = []
            for item in plan["subquestions"]:
                queries.append(item["question"])
            if not queries:
                queries.append(case["input"]["question"])
            raw_by_key = {}
            fused_by_key = {}
            reranked_by_key = {}
            for query in queries:
                dense = search_chunks_dense(
                    query, semantic_index, embed_model.embed, limit=SOURCE_LIMIT
                )
                sparse = search_chunks_sparse(
                    query, semantic_index, embed_model.embed, limit=SOURCE_LIMIT
                )
                sources = (("dense", dense), ("sparse", sparse))
                for _, values in sources:
                    for item in values:
                        raw_by_key[_key(item)] = item
                for item in fuse_ranked_chunks(sources, limit=SOURCE_LIMIT):
                    fused_by_key[_key(item)] = item
                for item in rerank_ranked_chunks(
                    query,
                    sources,
                    reranker=reranker,
                    limit=FINAL_LIMIT,
                ):
                    reranked_by_key[_key(item)] = item
            raw = tuple(raw_by_key.values())
            fused = tuple(fused_by_key.values())
            reranked = tuple(reranked_by_key.values())
            expected = _expected(case)
            coverable = []
            for item in expected:
                item_is_coverable = False
                for chunk in chunks:
                    if _covers(chunk, item):
                        item_is_coverable = True
                        break
                coverable.append(item_is_coverable)
            coverable_count = 0
            for was_coverable in coverable:
                if was_coverable:
                    coverable_count += 1
            rows.append(
                {
                    "case_id": case["id"],
                    "repository_id": repository_id,
                    "queries": queries,
                        "expected_evidence_count": len(expected),
                        "annotation_single_chunk_coverable_rate": (
                        coverable_count / len(coverable) if coverable else 1.0
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

    by_repository = {}
    for repository_id in repositories:
        repository_rows = []
        for row in rows:
            if row["repository_id"] == repository_id:
                repository_rows.append(row)
        by_repository[repository_id] = _summarize(repository_rows)

    payload = {
        "schema_version": 1,
        "case_count": len(rows),
        "summary": _summarize(rows),
        "by_repository": by_repository,
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
