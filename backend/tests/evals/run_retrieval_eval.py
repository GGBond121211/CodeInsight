"""Run lexical and BM25 retrieval evaluations and print a comparison."""

import json
from dataclasses import asdict
from pathlib import Path

from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.bm25 import search_chunks_bm25
from codeinsight.retrieval.lexical import search_chunks as search_chunks_lexical
from codeinsight.retrieval.metrics import evaluate_cases

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "cases.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"


def _evidence_items(case: dict) -> tuple[dict, ...]:
    return tuple(case.get("expected", {}).get("evidence", ()))


def _covers(chunk, evidence: dict) -> bool:
    return (
        chunk.relative_path == evidence["path"]
        and chunk.start_line <= evidence["start_line"]
        and chunk.end_line >= evidence["end_line"]
    )


def _coverage_rank(results, evidence: tuple[dict, ...]) -> int | None:
    covered = [False] * len(evidence)
    for result in results:
        for index, item in enumerate(evidence):
            covered[index] = covered[index] or _covers(result.chunk, item)
        if all(covered):
            return result.rank
    return None


def _case_comparison(cases: list[dict], chunks) -> list[dict]:
    comparisons: list[dict] = []
    for case in cases:
        evidence = _evidence_items(case)
        if not evidence:
            comparisons.append(
                {
                    "case_id": case["id"],
                    "applicable": False,
                    "reason": "insufficient_evidence",
                }
            )
            continue

        lexical_results = search_chunks_lexical(case["input"]["question"], chunks, limit=5)
        bm25_results = search_chunks_bm25(case["input"]["question"], chunks, limit=5)
        lexical_rank = _coverage_rank(lexical_results, evidence)
        bm25_rank = _coverage_rank(bm25_results, evidence)
        comparisons.append(
            {
                "case_id": case["id"],
                "applicable": True,
                "lexical_rank": lexical_rank,
                "bm25_rank": bm25_rank,
                "lexical_recall_at_5": lexical_rank is not None,
                "bm25_recall_at_5": bm25_rank is not None,
                "rank_change": (
                    None if lexical_rank is None or bm25_rank is None else lexical_rank - bm25_rank
                ),
            }
        )
    return comparisons


def _failed_case_ids(cases: list[dict], chunks, retrieval_mode: str) -> list[str]:
    failed: list[str] = []
    search = search_chunks_lexical if retrieval_mode == "lexical" else search_chunks_bm25
    for case in cases:
        evidence = _evidence_items(case)
        if not evidence:
            continue
        results = search(case["input"]["question"], chunks, limit=5)
        covers_all = True
        for evidence_item in evidence:
            evidence_is_covered = False
            for result in results:
                if _covers(result.chunk, evidence_item):
                    evidence_is_covered = True
                    break
            if not evidence_is_covered:
                covers_all = False
                break
        if not covers_all:
            failed.append(case["id"])
    return failed


def _metric_payload(metrics, *, retrieval_mode: str, parameters: dict | None = None) -> dict:
    payload = {"retrieval_mode": retrieval_mode, **asdict(metrics)}
    if parameters:
        payload["parameters"] = parameters
    return payload


def main() -> int:
    """Scan the fixture, evaluate both retrievers, and write comparison artifacts."""
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    scan_result = scan_repository(FIXTURE_ROOT)
    chunks = chunk_scan_result(scan_result)
    cases = document["cases"]
    lexical_metrics = evaluate_cases(cases, chunks, retrieval_mode="lexical")
    bm25_metrics = evaluate_cases(cases, chunks, retrieval_mode="bm25")
    lexical_payload = _metric_payload(
        lexical_metrics,
        retrieval_mode="lexical",
    )
    lexical_payload["failed_case_ids"] = _failed_case_ids(cases, chunks, "lexical")
    lexical_payload["notes"] = [
        "Frozen lexical baseline for the Project-0008 retrieval comparison.",
    ]
    bm25_payload = _metric_payload(
        bm25_metrics,
        retrieval_mode="bm25",
        parameters={"k1": 1.2, "b": 0.75},
    )
    bm25_payload["failed_case_ids"] = _failed_case_ids(cases, chunks, "bm25")
    bm25_payload["notes"] = [
        "BM25 body scoring with code-retrieval field bonuses.",
        "No embedding, vector retrieval, query rewriting, or synonym table was used.",
    ]
    delta = {}
    for metric in ("top1", "mean_reciprocal_rank", "recall_at_5", "valid_evidence_rate"):
        delta[metric] = bm25_payload[metric] - lexical_payload[metric]
    comparison = {
        "case_count": lexical_payload["case_count"],
        "applicable_case_count": lexical_payload["applicable_case_count"],
        "lexical": lexical_payload,
        "bm25": bm25_payload,
        "delta_bm25_minus_lexical": delta,
        "case_comparison": _case_comparison(cases, chunks),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "bm25-baseline.json").write_text(
        json.dumps(bm25_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "retrieval-comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
