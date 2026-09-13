"""Q-012 C2.1：对话入口分类器（task_type）在标注集上的表现。

这个分类器是纯规则函数，不调用模型，所以这份评测是**零成本**的：它可以随
代码一起重复跑，也可以作为回归护栏。

口径：
  - 逐条比较 expected 与 classify_chat_task 的返回值，不做「近似正确」；
  - core 与 boundary 分开统计：交界处的题本来就难，混在一起会掩盖两边的事实；
  - 同时输出混淆矩阵，因为「错成哪一类」比「错了」更有信息量——把 change 判成
    explain 是漏掉写门禁，把 scope_redirect 判成 explain 只是多跑一次检索。

用法：
    backend/.venv/Scripts/python.exe experiments/q12_router_eval.py
    加 --output 可换输出路径（默认 work/q12_router_eval.json）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.application.conversation_router import (  # noqa: E402
    classify_chat_task,
)

GOLDEN = ROOT / "experiments" / "configs" / "q12_router_golden.yaml"


def _load_cases() -> list[dict[str, object]]:
    document = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"标注集为空：{GOLDEN}")
    return [dict(item) for item in cases]


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for row in rows if row["expected"] == row["predicted"])
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q-012 C2.1 Router 分类准确率")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "q12_router_eval.json")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for case in _load_cases():
        question = str(case["question"])
        classification = classify_chat_task(
            question, has_active_code_goal=bool(case.get("active_goal", False))
        )
        rows.append(
            {
                "question": question,
                "expected": str(case["expected"]),
                "predicted": classification.task_type,
                "rule": classification.rule,
                "confidence": classification.confidence,
                "kind": str(case.get("kind", "core")),
                "ok": classification.task_type == str(case["expected"]),
            }
        )

    core = [row for row in rows if row["kind"] == "core"]
    boundary = [row for row in rows if row["kind"] == "boundary"]
    by_class: dict[str, dict[str, object]] = {}
    for name in sorted({str(row["expected"]) for row in rows}):
        subset = [row for row in rows if row["expected"] == name]
        by_class[name] = _summary(subset)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        confusion[str(row["expected"])][str(row["predicted"])] += 1

    category_counts = Counter(str(row["predicted"]) for row in rows)
    result = {
        "version": "q012-router-eval-v1",
        "golden": str(args.golden.relative_to(ROOT)).replace("\\", "/"),
        "overall": _summary(rows),
        "core": _summary(core),
        "boundary": _summary(boundary),
        "per_class": by_class,
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "predicted_distribution": dict(category_counts),
        "misses": [row for row in rows if not row["ok"]],
        "records": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"总准确率：{result['overall']['correct']}/{result['overall']['total']} "
          f"= {result['overall']['accuracy']:.3f}")
    print(f"core：{core_summary_bits(result['core'])}")
    print(f"boundary：{core_summary_bits(result['boundary'])}")
    for name, item in by_class.items():
        print(f"  {name:16s} {item['correct']}/{item['total']} = {item['accuracy']:.3f}")
    print("混淆矩阵（行=期望，列=实际）：")
    for expected, counts in result["confusion"].items():
        pairs = "、".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        print(f"  {expected} -> {pairs}")
    if result["misses"]:
        print("判错的条目：")
        for row in result["misses"]:
            print(f"  [{row['kind']}] {row['question']} -> {row['predicted']} "
                  f"(期望 {row['expected']}，命中规则 {row['rule']})")
    print(f"已写入 {args.output}")
    return 0


def core_summary_bits(item: dict[str, object]) -> str:
    return f"{item['correct']}/{item['total']} = {float(item['accuracy']):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
