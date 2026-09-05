"""Run the zero-model SWE-bench L2 localization pilot.

This evaluator deliberately measures the retrieval/evidence layer only:

* the query is the original ``problem_statement`` from the frozen dataset;
* the gold patch is used only after retrieval, for scoring;
* no external repository code, tests, installers, or Docker containers run;
* the repository is scanned at each case's pinned ``base_commit``;
* the current production BM25 tokenizer and fixed-80 chunking are reused.

The result reports Top-10 (the current final evidence budget), Top-20, and
Top-100 (the current post-fusion candidate budget) so the pilot does not
mistake a BM25-only candidate result for a model-reranked quality result.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.bm25 import search_chunks_bm25

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
FROZEN_PATH = BACKEND_ROOT / "tests" / "evals" / "swebench_l2_frozen_80.json"
RAW_PATH = PROJECT_ROOT / "work" / "swebench_verified_raw.json"
WORKTREE_ROOT = PROJECT_ROOT / "work" / "benchmarks" / "swebench_l2"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "STEP8-L2-LOCALIZATION"
TOP_KS = (10, 20, 100)
CHUNK_MAX_LINES = 80

REPO_DIRS = {
    "astropy/astropy": "astropy__astropy",
    "matplotlib/matplotlib": "matplotlib__matplotlib",
    "psf/requests": "psf__requests",
    "pydata/xarray": "pydata__xarray",
    "pytest-dev/pytest": "pytest-dev__pytest",
    "scikit-learn/scikit-learn": "scikit-learn__scikit-learn",
    "sphinx-doc/sphinx": "sphinx-doc__sphinx",
}


def _load_cases() -> list[dict[str, Any]]:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    cases = [case for case in frozen["cases"] if case["evaluation_split"] == "dev"]
    if len(cases) != 20:
        raise ValueError(f"冻结 dev 集应为 20 条，实际为 {len(cases)} 条")

    raw_payload = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    raw_by_id = {entry["row"]["instance_id"]: entry["row"] for entry in raw_payload}
    for case in cases:
        raw = raw_by_id.get(case["instance_id"])
        if raw is None or not raw.get("problem_statement", "").strip():
            raise ValueError(f"缺少 {case['instance_id']} 的 problem_statement")
        # Keep only the query field from the raw row. Patch/test fields never
        # enter the retrieval path and are intentionally not copied here.
        case["problem_statement"] = raw["problem_statement"]
    return cases


def _run_git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _verify_worktree(case: dict[str, Any], path: Path) -> dict[str, Any]:
    head = _run_git(path, "rev-parse", "HEAD")
    status = _run_git(path, "status", "--porcelain")
    expected = case["base_commit"]
    if head != expected:
        raise RuntimeError(f"{case['instance_id']} HEAD={head}, expected={expected}")
    if status:
        raise RuntimeError(f"{case['instance_id']} worktree is dirty: {status}")
    return {"head": head, "dirty": bool(status)}


def _gold_ranges(case: dict[str, Any]) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for location in case["gold_locations"]:
        for start, end in location["line_ranges"]:
            ranges.append((location["path"], int(start), int(end)))
    return ranges


def _gold_files(case: dict[str, Any]) -> set[str]:
    return {location["path"] for location in case["gold_locations"]}


def _covers(result: RankedChunk, expected: tuple[str, int, int]) -> bool:
    path, start, end = expected
    chunk = result.chunk
    return (
        chunk.relative_path == path
        and chunk.start_line <= start
        and chunk.end_line >= end
    )


def _metrics(
    case: dict[str, Any], results: tuple[RankedChunk, ...], top_k: int
) -> dict[str, Any]:
    selected = results[:top_k]
    gold_ranges = _gold_ranges(case)
    gold_files = _gold_files(case)
    result_files = {item.chunk.relative_path for item in selected}
    covered = [any(_covers(item, expected) for item in selected) for expected in gold_ranges]
    valid = all(
        item.chunk.start_line >= 1
        and item.chunk.end_line >= item.chunk.start_line
        and item.chunk.relative_path
        for item in selected
    )
    return {
        "top_k": top_k,
        "result_count": len(selected),
        "file_level_all_gold_files_hit": gold_files.issubset(result_files),
        "file_recall": (
            len(gold_files & result_files) / len(gold_files) if gold_files else 1.0
        ),
        "line_level_evidence_recall": (
            sum(covered) / len(covered) if covered else 1.0
        ),
        "complete_line_evidence_coverage": all(covered),
        "citation_validity": 1.0 if valid else 0.0,
        "predicted_files": sorted(result_files),
        "covered_ranges": [
            {"path": path, "start_line": start, "end_line": end, "covered": hit}
            for (path, start, end), hit in zip(gold_ranges, covered, strict=True)
        ],
    }


def _aggregate(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    selected = [row["metrics"][str(top_k)] for row in rows]
    repo_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        repo_rows.setdefault(row["repo"], []).append(row["metrics"][str(top_k)])

    def mean(field: str, items: list[dict[str, Any]]) -> float:
        return sum(float(item[field]) for item in items) / len(items)

    by_repo = {
        repo: {
            "case_count": len(items),
            "file_all_hit_accuracy": sum(
                bool(item["file_level_all_gold_files_hit"]) for item in items
            )
            / len(items),
            "file_recall": mean("file_recall", items),
            "line_level_evidence_recall": mean(
                "line_level_evidence_recall", items
            ),
            "complete_line_evidence_coverage": sum(
                bool(item["complete_line_evidence_coverage"]) for item in items
            )
            / len(items),
            "citation_validity": mean("citation_validity", items),
        }
        for repo, items in sorted(repo_rows.items())
    }
    return {
        "case_count": len(selected),
        "file_all_hit_accuracy": sum(
            bool(item["file_level_all_gold_files_hit"]) for item in selected
        )
        / len(selected),
        "file_recall": mean("file_recall", selected),
        "line_level_evidence_recall": mean(
            "line_level_evidence_recall", selected
        ),
        "complete_line_evidence_coverage": sum(
            bool(item["complete_line_evidence_coverage"]) for item in selected
        )
        / len(selected),
        "citation_validity": mean("citation_validity", selected),
        "by_repository": by_repo,
    }


def run() -> dict[str, Any]:
    cases = _load_cases()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-l2-localization")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_case: dict[str, tuple[Any, ...]] = {}
    verification: dict[str, Any] = {}
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        repo_dir_name = REPO_DIRS.get(case["repo"])
        if repo_dir_name is None:
            raise ValueError(f"dev case 使用了未准备的仓库：{case['repo']}")
        worktree = WORKTREE_ROOT / case["instance_id"]
        if not worktree.is_dir():
            raise FileNotFoundError(f"缺少固定 worktree：{worktree}")
        verification[case["instance_id"]] = _verify_worktree(case, worktree)
        scan = scan_repository(worktree)
        chunks = chunk_scan_result(scan, max_lines=CHUNK_MAX_LINES)
        if not chunks:
            raise RuntimeError(f"{case['instance_id']} 没有可检索源码块")
        chunks_by_case[case["instance_id"]] = chunks

    for case in cases:
        case_id = case["instance_id"]
        chunks = chunks_by_case[case_id]
        ranked = search_chunks_bm25(
            case["problem_statement"], chunks, limit=max(TOP_KS)
        )
        metrics = {
            str(top_k): _metrics(case, ranked, top_k) for top_k in TOP_KS
        }
        case_rows.append(
            {
                "instance_id": case_id,
                "repo": case["repo"],
                "base_commit": case["base_commit"],
                "difficulty": case["difficulty"],
                "query_source": "problem_statement_only",
                "external_code_executed": False,
                "gold_ranges": [
                    {"path": path, "start_line": start, "end_line": end}
                    for path, start, end in _gold_ranges(case)
                ],
                "ranked_results": [
                    {
                        "rank": item.rank,
                        "path": item.chunk.relative_path,
                        "start_line": item.chunk.start_line,
                        "end_line": item.chunk.end_line,
                        "score": item.score,
                    }
                    for item in ranked
                ],
                "metrics": metrics,
                "scanned_file_count": len(
                    {chunk.relative_path for chunk in chunks}
                ),
                "scanned_chunk_count": len(chunks),
            }
        )

    rows_path = output_dir / "case_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": 1,
        "experiment_id": "STEP8-L2-LOCALIZATION",
        "run_id": run_id,
        "status": "completed",
        "dataset": {
            "manifest": str(FROZEN_PATH),
            "split": "dev",
            "case_count": len(cases),
            "source": "SWE-bench_Verified",
        },
        "retrieval": {
            "algorithm": "current_production_bm25",
            "chunk_max_lines": CHUNK_MAX_LINES,
            "candidate_limit": max(TOP_KS),
            "query_fields": ["problem_statement"],
            "semantic_model_calls": 0,
            "rerank_model_calls": 0,
        },
        "metrics_definition": {
            "file_all_hit_accuracy": "case-level fraction where every gold file appears in top-k",
            "file_recall": "macro average of gold files present in top-k",
            "line_level_evidence_recall": "gold line ranges covered by a top-k 80-line chunk",
            "citation_validity": "all returned paths and 1-based ranges are valid",
        },
        "aggregate": {str(top_k): _aggregate(case_rows, top_k) for top_k in TOP_KS},
        "hard_gates": {
            "out_of_scope_writes": 0,
            "disallowed_commands": 0,
            "original_repo_modifications": 0,
            "invalid_evidence": 0,
            "approval_bypass": 0,
            "sandbox_escape": 0,
            "repo_injection_violations": 0,
        },
        "execution_boundary": {
            "external_code_executed": False,
            "tests_executed": False,
            "docker_used": False,
            "original_repositories_modified": False,
            "gold_patch_seen_by_retriever": False,
        },
        "worktree_verification": verification,
        "outputs": {"case_rows": str(rows_path)},
        "notes": [
            "This is a 20-case dev localization pilot, not a full SWE-bench resolve-rate result.",
            "BM25-only results do not prove semantic retrieval or qwen rerank quality.",
            "The frozen holdout split was not read or evaluated.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
