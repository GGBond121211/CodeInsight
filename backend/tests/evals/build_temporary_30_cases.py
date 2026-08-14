"""Build the bounded 30-case high-difficulty smoke asset for dense iteration."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SOURCE_PATH = EVAL_DIR / "multilingual_cases.json"
OUTPUT_PATH = EVAL_DIR / "multilingual_cases_temporary_30.json"

# Deliberately cover every v2 category, both language forms, and all four
# expression variants instead of selecting only the easiest or most common
# cases.  These IDs are a stable reviewable sampling contract for dense runs.
SELECTED_CASE_IDS = (
    "t11-v2-record-symbol-collision-a",
    "t11-v2-eta-sla-collision-b",
    "t11-v2-state-name-collision-c",
    "t11-v2-reserve-boundary-b",
    "t11-v2-ledger-persistence-d",
    "t11-v2-dispatch-priority-c",
    "t11-v2-docs-carrier-conflict-b",
    "t11-v2-docs-record-conflict-c",
    "t11-v2-docs-record-conflict-d",
    "t11-v2-checkout-full-trace-a",
    "t11-v2-checkout-full-trace-b",
    "t11-v2-fulfillment-return-chain-c",
    "t11-v2-checkout-failure-order-d",
    "t11-v2-response-field-lineage-a",
    "t11-v2-response-field-lineage-b",
    "t11-v2-quantity-boundary-c",
    "t11-v2-total-calculation-d",
    "t11-v2-missing-payment-provider-a",
    "t11-v2-missing-http-route-b",
    "t11-v2-multi-checkout-claims-a",
    "t11-v2-multi-checkout-claims-b",
    "t11-v2-multi-record-refund-c",
    "t11-v2-multi-shipping-concepts-d",
    "t11-v2-validation-typo-a",
    "t11-v2-money-currency-typos-b",
    "t11-v2-fulfillment-noisy-nesting-c",
    "t11-v2-quarantine-paraphrase-a",
    "t11-v2-quarantine-paraphrase-b",
    "t11-v2-fragile-paraphrase-c",
    "t11-v2-currency-paraphrase-d",
)


def build_document() -> dict:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in source["cases"]}
    missing = [case_id for case_id in SELECTED_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"selected temporary cases are missing: {missing}")
    cases = [deepcopy(by_id[case_id]) for case_id in SELECTED_CASE_IDS]
    if len(cases) != 30 or len({case["id"] for case in cases}) != 30:
        raise ValueError("temporary asset must contain 30 unique cases")
    return {
        "schema_version": 1,
        "case_set": "task11_multilingual_temporary_30",
        "fixture": source["fixture"],
        "description": (
            "A reviewable 30-case temporary subset of the 100-case Task 11 v2 asset "
            "for dense implementation iteration. It is not a replacement for v2 or Fusion-131."
        ),
        "selection_contract": {
            "source_asset": "tests/evals/multilingual_cases.json",
            "source_case_set": source["case_set"],
            "selection": (
                "hand-curated coverage of every category, both zh/zh-en forms, and hard variants"
            ),
            "case_count": 30,
            "categories": dict(sorted(Counter(case["category"] for case in cases).items())),
            "languages": dict(sorted(Counter(case["language"] for case in cases).items())),
            "preserves_original_cases": True,
        },
        "cases": cases,
    }


def main() -> int:
    document = build_document()
    OUTPUT_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH),
                "case_count": len(document["cases"]),
                "categories": document["selection_contract"]["categories"],
                "languages": document["selection_contract"]["languages"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
