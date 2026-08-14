"""Merge completed fusion answer-evaluation batches without making model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_fusion_answer_eval import grouped_summary, summarize

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "multilingual_cases_fusion.json"
OUTPUT_ROOT = BACKEND_ROOT.parent / "outputs" / "evals"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("linear", "router-linear", "router-agent"), required=True
    )
    args = parser.parse_args()
    pattern = f"multilingual-fusion-answer-comparison-{args.mode}-batch*.json"
    paths = sorted(OUTPUT_ROOT.glob(pattern))
    if not paths:
        parser.error(f"no batch artifacts match {pattern}")

    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    expected_cases = document["cases"]
    expected_ids = [case["id"] for case in expected_cases]
    batches = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results = [item for batch in batches for item in batch.get("cases", [])]
    result_ids = [item["case_id"] for item in results]
    if len(result_ids) != len(set(result_ids)):
        parser.error("batch artifacts contain duplicate case IDs")
    if set(result_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(expected_ids))
        parser.error(f"case ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    by_id = {item["case_id"]: item for item in results}
    ordered = [by_id[case_id] for case_id in expected_ids]
    first = batches[0]
    payload = {
        "schema_version": 1,
        "case_set": document["case_set"],
        "mode": args.mode,
        "model": first.get("model"),
        "embedding_model_configured": any(
            batch.get("embedding_model_configured") for batch in batches
        ),
        "base_url_host": first.get("base_url_host"),
        "batch_count": len(batches),
        "batches": [path.name for path in paths],
        "summary": summarize(ordered),
        "groups": grouped_summary(ordered),
        "sources": document["sources"],
        "notes": [
            "This artifact is merged from completed per-batch runs.",
            "All 131 cases are present exactly once in source-asset order.",
            "No additional model or embedding calls are made by the merger.",
            "No LLM judge is used; outcome and citation metrics are deterministic.",
        ],
        "cases": ordered,
    }
    output_path = OUTPUT_ROOT / f"multilingual-fusion-answer-comparison-{args.mode}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output_path), "summary": payload["summary"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
