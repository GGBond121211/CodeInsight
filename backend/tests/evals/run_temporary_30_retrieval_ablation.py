"""Run A0-A5 retrieval ablations for the frozen Temporary-30 manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from time import perf_counter

from codeinsight.application.search_repository import (
    build_repository_semantic_index,
    search_repository,
)
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.retrieval.hybrid import fuse_ranked_chunks, rerank_ranked_chunks
from codeinsight.retrieval.semantic import search_chunks_semantic

try:
    from .temporary_30_ablation import (
        FIXTURE_ROOT,
        MANIFEST_PATH,
        RETRIEVAL_GROUPS,
        environment_fingerprint,
        git_state,
        run_directory,
        selected_cases,
        sha256_file,
        write_json_new,
    )
except ImportError:  # Direct script execution.
    from temporary_30_ablation import (
        FIXTURE_ROOT,
        MANIFEST_PATH,
        RETRIEVAL_GROUPS,
        environment_fingerprint,
        git_state,
        run_directory,
        selected_cases,
        sha256_file,
        write_json_new,
    )

SPARSE_MODES = ("bm25",)
SOURCE_LIMIT = 20
FINAL_LIMIT = 5


def _key(result) -> tuple[str, int, int]:
    return (result.chunk.relative_path, result.chunk.start_line, result.chunk.end_line)


def _covers(result, expected: dict) -> bool:
    return (
        result.chunk.relative_path == expected["path"]
        and result.chunk.start_line <= expected["start_line"]
        and result.chunk.end_line >= expected["end_line"]
    )


def _candidate_payload(results, source_ranks: dict) -> list[dict]:
    return [
        {
            "path": result.chunk.relative_path,
            "start_line": result.chunk.start_line,
            "end_line": result.chunk.end_line,
            "rank": result.rank,
            "score": result.score,
            "retrieval_reason": result.retrieval_reason,
            "found_by": sorted(source_ranks.get(_key(result), {})),
            "source_ranks": source_ranks.get(_key(result), {}),
        }
        for result in results
    ]


def _source_rank_map(sources) -> dict:
    mapping: defaultdict[tuple[str, int, int], dict[str, int]] = defaultdict(dict)
    for source_name, results in sources:
        for result in results:
            mapping[_key(result)][source_name] = result.rank
    return dict(mapping)


def _metrics(expected: list[dict], raw_union, fused, final) -> dict:
    if not expected:
        return {
            "expected_evidence_count": 0,
            "raw_source_candidate_recall": None,
            "fused_candidate_recall_at_20": None,
            "evidence_recall_at_5": None,
            "complete_evidence_coverage_at_5": None,
            "mrr": None,
            "top1": None,
            "fusion_retention": None,
            "conditional_rerank_success": None,
        }
    raw_hits = [any(_covers(item, evidence) for item in raw_union) for evidence in expected]
    fused_hits = [any(_covers(item, evidence) for item in fused) for evidence in expected]
    final_hits = [any(_covers(item, evidence) for item in final) for evidence in expected]
    first_rank = next(
        (
            result.rank
            for result in final
            if any(_covers(result, evidence) for evidence in expected)
        ),
        None,
    )
    return {
        "expected_evidence_count": len(expected),
        "raw_source_candidate_recall": sum(raw_hits) / len(expected),
        "fused_candidate_recall_at_20": sum(fused_hits) / len(expected),
        "evidence_recall_at_5": sum(final_hits) / len(expected),
        "complete_evidence_coverage_at_5": all(final_hits),
        "mrr": 1 / first_rank if first_rank else 0.0,
        "top1": first_rank == 1,
        "fusion_retention": (any(fused_hits) if any(raw_hits) else None),
        "conditional_rerank_success": (any(final_hits) if any(fused_hits) else None),
    }


def _path_line_valid(results) -> bool:
    for result in results:
        path = FIXTURE_ROOT / result.chunk.relative_path
        if not path.is_file():
            return False
        lines = path.read_text(encoding="utf-8").splitlines()
        if not (1 <= result.chunk.start_line <= result.chunk.end_line <= len(lines)):
            return False
    return True


def evaluate_subquestion(
    subquestion: dict,
    *,
    semantic_index,
    embed,
) -> dict:
    question = subquestion["question"]
    sources = {
        mode: search_repository(
            FIXTURE_ROOT,
            question,
            limit=SOURCE_LIMIT,
            retrieval_mode=mode,
        )
        for mode in SPARSE_MODES
    }
    sources["semantic"] = search_chunks_semantic(
        question, semantic_index, embed, limit=SOURCE_LIMIT
    )
    groups = {
        "A0": (subquestion["primary_sparse_mode"],),
        "A1": SPARSE_MODES,
        "A2": SPARSE_MODES,
        "A3": ("semantic",),
        "A4": (*SPARSE_MODES, "semantic"),
        "A5": (*SPARSE_MODES, "semantic"),
    }
    results = {}
    for group, source_names in groups.items():
        group_sources = tuple((name, sources[name]) for name in source_names)
        source_ranks = _source_rank_map(group_sources)
        candidates_by_key = {_key(item): item for _, values in group_sources for item in values}
        raw_union = tuple(candidates_by_key[key] for key in sorted(source_ranks))
        if group in {"A0", "A3"}:
            fused = tuple(group_sources[0][1][:SOURCE_LIMIT])
            final = tuple(group_sources[0][1][:FINAL_LIMIT])
        else:
            fused = fuse_ranked_chunks(group_sources, limit=SOURCE_LIMIT)
            final = (
                rerank_ranked_chunks(question, group_sources, limit=FINAL_LIMIT)
                if group in {"A2", "A5"}
                else tuple(fused[:FINAL_LIMIT])
            )
        results[group] = {
            "name": "RRF + agreement" if group in {"A1", "A4"} else group,
            "sources": list(source_names),
            "raw_union_count": len(raw_union),
            "fused_top20": _candidate_payload(fused, source_ranks),
            "final_top5": _candidate_payload(final, source_ranks),
            "path_line_valid": _path_line_valid(final),
            **_metrics(subquestion["expected_evidence"], raw_union, fused, final),
        }
    return {
        "subquestion_id": subquestion["id"],
        "question": question,
        "expected_outcome": subquestion["expected_outcome"],
        "raw_sources": {
            name: _candidate_payload(values, _source_rank_map(((name, values),)))
            for name, values in sources.items()
        },
        "groups": results,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for group in RETRIEVAL_GROUPS:
        items = [row["groups"][group] for row in rows]
        evidence_items = [item for item in items if item["expected_evidence_count"]]
        summary[group] = {
            "name": items[0]["name"] if items else group,
            "subquestion_count": len(items),
            "raw_source_candidate_recall": sum(
                item["raw_source_candidate_recall"] for item in evidence_items
            )
            / len(evidence_items),
            "fused_candidate_recall_at_20": sum(
                item["fused_candidate_recall_at_20"] for item in evidence_items
            )
            / len(evidence_items),
            "evidence_recall_at_5": sum(item["evidence_recall_at_5"] for item in evidence_items)
            / len(evidence_items),
            "complete_evidence_coverage_at_5": sum(
                item["complete_evidence_coverage_at_5"] for item in evidence_items
            )
            / len(evidence_items),
            "mrr": sum(item["mrr"] for item in evidence_items) / len(evidence_items),
            "top1": sum(item["top1"] for item in evidence_items) / len(evidence_items),
            "path_line_validity": sum(item["path_line_valid"] for item in items) / len(items),
        }
    return summary


def build_retrieval_payload(split: str, embed) -> dict:
    cases = selected_cases(split)
    started = perf_counter()
    semantic_index = build_repository_semantic_index(FIXTURE_ROOT, semantic_embed=embed)
    rows = []
    errors = []
    for case in cases:
        for subquestion in case["subquestions"]:
            try:
                result = evaluate_subquestion(
                    subquestion, semantic_index=semantic_index, embed=embed
                )
                result.update(
                    {
                        "case_id": case["case_id"],
                        "language": case["language"],
                        "category": case["category"],
                    }
                )
                rows.append(result)
            except Exception as error:  # noqa: BLE001 - evaluator records boundaries
                errors.append(
                    {
                        "case_id": case["case_id"],
                        "subquestion_id": subquestion["id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return {
        "schema_version": 1,
        "split": split,
        "case_count": len(cases),
        "subquestion_count": sum(len(case["subquestions"]) for case in cases),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "semantic_index_id": semantic_index.metadata.index_id,
        "embedding_model": semantic_index.metadata.model,
        "elapsed_milliseconds": (perf_counter() - started) * 1000,
        "summary": summarize(rows) if rows else {},
        "errors": errors,
        "subquestions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("diagnostic", "confirmation"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    cases = selected_cases(args.split)
    budget = {
        "split": args.split,
        "case_count": len(cases),
        "subquestion_count": sum(len(case["subquestions"]) for case in cases),
        "embedding_index_batches": 1,
        "embedding_query_calls": sum(len(case["subquestions"]) for case in cases),
        "environment": environment_fingerprint(),
    }
    print(json.dumps(budget, ensure_ascii=False))
    if not args.confirm_run:
        return 0
    if not os.environ.get("CODEINSIGHT_EMBEDDING_MODEL"):
        parser.error("CODEINSIGHT_EMBEDDING_MODEL is required")
    output_dir = run_directory(args.run_id, args.split)
    output_path = output_dir / "retrieval-ablation.json"
    if output_path.exists():
        parser.error(f"output already exists: {output_path}")
    model = OpenAIEmbeddingModel.from_environment()
    payload = build_retrieval_payload(args.split, model.embed)
    payload["git"] = git_state()
    payload["environment"] = environment_fingerprint()
    write_json_new(output_path, payload)
    print(
        json.dumps(
            {"output": str(output_path), "summary": payload["summary"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
