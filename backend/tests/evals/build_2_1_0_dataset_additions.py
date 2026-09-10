"""检查 2.1.0 多轮数据集的新增样本清单。

新增案例是经过源码审查的固定 JSON，不由模型随机生成。这个脚本的职责是
验证 addition manifest 与主数据集一致，并输出稳定摘要；它不调用 Provider。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
DATASET_PATH = EVAL_ROOT / "conversation_cases_2_1_0.json"
ADDITIONS_PATH = EVAL_ROOT / "dataset_additions_2_1_0.json"


def build_summary(
    dataset_path: Path = DATASET_PATH, additions_path: Path = ADDITIONS_PATH
) -> dict[str, object]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    additions = json.loads(additions_path.read_text(encoding="utf-8"))
    cases_by_id = {case["id"]: case for case in dataset["cases"]}
    case_ids = additions["added_case_ids"]
    missing = [case_id for case_id in case_ids if case_id not in cases_by_id]
    turns = sum(
        len(cases_by_id[case_id]["turns"]) for case_id in case_ids if case_id in cases_by_id
    )
    return {
        "dataset_id": dataset["dataset_id"],
        "addition_id": additions["addition_id"],
        "added_case_count": len(case_ids),
        "declared_added_case_count": additions["added_case_count"],
        "added_turn_count": turns,
        "declared_added_turn_count": additions["added_turn_count"],
        "missing_case_ids": missing,
        "source_commit": additions["source_commit"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--additions", type=Path, default=ADDITIONS_PATH)
    args = parser.parse_args()
    summary = build_summary(args.dataset, args.additions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    valid = (
        not summary["missing_case_ids"]
        and summary["added_case_count"] == summary["declared_added_case_count"]
        and summary["added_turn_count"] == summary["declared_added_turn_count"]
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
