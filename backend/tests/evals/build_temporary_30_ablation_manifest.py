"""Build the frozen Temporary-30 ablation manifest from the reviewed asset."""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SOURCE_PATH = EVAL_DIR / "multilingual_cases_temporary_30.json"
OUTPUT_PATH = EVAL_DIR / "temporary_30_ablation_manifest.json"

CONFIRMATION_IDS = frozenset(
    {
        "t11-v2-record-symbol-collision-a",
        "t11-v2-reserve-boundary-b",
        "t11-v2-docs-carrier-conflict-b",
        "t11-v2-fulfillment-return-chain-c",
        "t11-v2-quantity-boundary-c",
        "t11-v2-missing-http-route-b",
        "t11-v2-multi-record-refund-c",
        "t11-v2-money-currency-typos-b",
        "t11-v2-fragile-paraphrase-c",
        "t11-v2-multi-shipping-concepts-d",
    }
)

PRIMARY_MODE_BY_CATEGORY = {
    "ambiguity": "bm25",
    "boundary_behavior": "bm25",
    "code_document_conflict": "bm25",
    "cross_file_trace": "bm25",
    "data_flow": "bm25",
    "insufficient_evidence": "bm25",
    "multi_intent": "bm25",
    "noisy_query": "bm25",
    "semantic_paraphrase": "bm25",
}

MULTI_CLAIMS = {
    "API 是否构建 OrderRequest？": ("API constructs an OrderRequest before checkout.", "true"),
    "校验是否发生在库存预留之后？": (
        "Validation occurs before inventory reservation, so the premise is false.",
        "false",
    ),
    "总价是否包含税或折扣？": (
        "The total is unit price times quantity without tax or discounts.",
        "false",
    ),
    "customer lookup 是否校验 owner？": (
        "Customer lookup rejects a row owned by another customer.",
        "true",
    ),
    "admin lookup 返回什么 scope？": ("Admin lookup returns scope admin.", "true"),
    "queued refund 是否调用 Stripe？": (
        "The repository does not establish a Stripe call or settlement.",
        "unsupported",
    ),
    "易碎国内件选择哪条 lane？": (
        "Fragile takes priority and selects fragile-courier.",
        "true",
    ),
    "ETA 与 SLA 分别如何计算？": (
        "ETA adds transit days while SLA adds response hours to different base times.",
        "true",
    ),
    "finalize_shipment 是否调用外部承运商？": (
        "The repository does not establish an external carrier call.",
        "unsupported",
    ),
}


def _subquestions(case: dict) -> list[dict]:
    expected = case["expected"]
    raw_subquestions = expected.get("subquestions")
    if raw_subquestions:
        return raw_subquestions
    return [
        {
            "question": case["input"]["question"],
            "outcome": expected["outcome"],
            "evidence": expected.get("evidence", []),
        }
    ]


def build_manifest() -> dict:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    cases = []
    claim_counter = 1
    for case in source["cases"]:
        subquestions = []
        for index, subquestion in enumerate(_subquestions(case), start=1):
            outcome = subquestion["outcome"]
            if subquestion["question"] in MULTI_CLAIMS:
                statement, polarity = MULTI_CLAIMS[subquestion["question"]]
            elif outcome == "insufficient_evidence":
                statement = (
                    "The requested external integration or deployment fact is not established "
                    "by this repository."
                )
                polarity = "unsupported"
            else:
                statement = case["protects"]
                polarity = "true"
            subquestions.append(
                {
                    "id": f"Q{index}",
                    "question": subquestion["question"],
                    "intent": case["category"],
                    "expected_outcome": outcome,
                    "primary_sparse_mode": PRIMARY_MODE_BY_CATEGORY[case["category"]],
                    "expected_evidence": (
                        []
                        if outcome == "insufficient_evidence"
                        else subquestion.get("evidence", [])
                    ),
                    "atomic_claims": [
                        {
                            "claim_id": f"C{claim_counter:03d}",
                            "expected_statement": statement,
                            "polarity": polarity,
                        }
                    ],
                }
            )
            claim_counter += 1
        cases.append(
            {
                "case_id": case["id"],
                "scenario": case["scenario"],
                "split": ("confirmation" if case["id"] in CONFIRMATION_IDS else "diagnostic"),
                "language": case["language"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "challenge_features": case["challenge_features"],
                "original_question": case["input"]["question"],
                "expected_case_outcome": case["expected"]["outcome"],
                "subquestions": subquestions,
            }
        )
    return {
        "schema_version": 1,
        "case_set": "temporary_30_hybrid_critic_ablation",
        "source_asset": SOURCE_PATH.name,
        "fixture": "tests/fixtures/sample_repo",
        "seed": 20260812,
        "case_count": len(cases),
        "subquestion_count": sum_subquestion_count(cases),
        "confirmation_ids": sorted(CONFIRMATION_IDS),
        "cases": cases,
    }


def sum_subquestion_count(cases: list[dict]) -> int:
    count = 0
    for case in cases:
        count += len(case["subquestions"])
    return count


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
