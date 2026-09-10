"""2.1.0 新增多轮样本的确定性回归测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_2_1_0_dataset_additions import build_summary

EVAL_ROOT = Path(__file__).resolve().parent


def test_additions_are_present_and_counted() -> None:
    summary = build_summary()

    assert summary["missing_case_ids"] == []
    assert summary["added_case_count"] == 4
    assert summary["added_turn_count"] == 19


def test_addition_ids_are_not_duplicated() -> None:
    additions = json.loads(
        (EVAL_ROOT / "dataset_additions_2_1_0.json").read_text(encoding="utf-8")
    )

    ids = additions["added_case_ids"]
    assert len(ids) == len(set(ids))
