"""从 194 条候选中冻结可复现的 SWE-bench L2 定位评测集。

本脚本不访问网络，也不修改原始候选清单。选择依据写进输出 manifest：

* 固定 80 条，处在计划允许的 50--100 范围中；
* 按仓库配额保留 10 个仓库，避免单仓库主导；
* 每个仓库内部再按 difficulty 分层；
* 用 instance_id 的 SHA-256 排序实现稳定、可回放的选样，而不是依赖
  当前文件顺序或一次性的随机种子；
* 再把 80 条分成 dev/regression/golden/holdout 四个 20 条 split。

L2 仍然只做 issue/gold patch 对齐的文件和行定位，不构建 Docker、不运行测试。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "swebench_l2_subset.json"
OUTPUT_PATH = HERE / "swebench_l2_frozen_80.json"

# 80/194 的比例分配后取整，并保证十个已选仓库都保留。
REPO_TARGETS: dict[str, int] = {
    "astropy/astropy": 9,
    "matplotlib/matplotlib": 14,
    "mwaskom/seaborn": 1,
    "pallets/flask": 1,
    "psf/requests": 3,
    "pydata/xarray": 9,
    "pylint-dev/pylint": 4,
    "pytest-dev/pytest": 8,
    "scikit-learn/scikit-learn": 13,
    "sphinx-doc/sphinx": 18,
}

SPLIT_TARGETS = {"dev": 20, "regression": 20, "golden": 20, "holdout": 20}


def stable_key(case: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"codeinsight-l2-freeze-v1:{case['instance_id']}".encode()
    ).hexdigest()


def proportional_quotas(
    counts: dict[str, int], total: int, *, minimum_one: bool = False
) -> dict[str, int]:
    """按最大余数法分配整数配额，避免顺序造成偏差。"""
    if total < 0 or total > sum(counts.values()):
        raise ValueError("配额超出可选样本范围")
    keys = sorted(counts)
    quotas = {key: 1 if minimum_one and counts[key] else 0 for key in keys}
    remaining = total - sum(quotas.values())
    if remaining < 0:
        raise ValueError("minimum_one 配额超过总数")
    available = {key: counts[key] - quotas[key] for key in keys}
    denominator = sum(counts.values())
    raw = {key: (remaining * available[key] / denominator) for key in keys}
    for key in keys:
        quotas[key] += int(raw[key])
    left = total - sum(quotas.values())
    ranked = sorted(
        keys,
        key=lambda key: (raw[key] - int(raw[key]), -available[key], key),
        reverse=True,
    )
    for key in ranked[:left]:
        quotas[key] += 1
    return quotas


def select_by_repo(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_repo[case["repo"]].append(case)
    if set(by_repo) != set(REPO_TARGETS):
        raise ValueError("候选仓库集合已变化，必须重新审阅配额")

    selected: list[dict[str, Any]] = []
    for repo, target in REPO_TARGETS.items():
        repo_cases = by_repo[repo]
        if target > len(repo_cases):
            raise ValueError(f"{repo} 只有 {len(repo_cases)} 条，无法取 {target} 条")
        by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in repo_cases:
            by_difficulty[case["difficulty"]].append(case)
        difficulty_counts = {key: len(value) for key, value in by_difficulty.items()}
        quotas = proportional_quotas(difficulty_counts, target)
        for difficulty in sorted(by_difficulty):
            bucket = sorted(by_difficulty[difficulty], key=stable_key)
            selected.extend(bucket[: quotas[difficulty]])
    return sorted(selected, key=lambda case: case["instance_id"])


def assign_splits(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按稳定 hash 的全局顺序分配四个等大 split。"""
    ordered = sorted(cases, key=stable_key)
    assigned: list[dict[str, Any]] = []
    boundaries: list[tuple[str, int]] = []
    cursor = 0
    for split, count in SPLIT_TARGETS.items():
        cursor += count
        boundaries.append((split, cursor))
    for index, case in enumerate(ordered, start=1):
        split = next(name for name, boundary in boundaries if index <= boundary)
        copy = dict(case)
        copy["evaluation_split"] = split
        assigned.append(copy)
    return sorted(assigned, key=lambda case: case["instance_id"])


def build_manifest() -> dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    candidates = source["cases"]
    selected = assign_splits(select_by_repo(candidates))
    if len(selected) != sum(SPLIT_TARGETS.values()):
        raise AssertionError("冻结数量与 split 配额不一致")

    repo_distribution = dict(sorted(Counter(case["repo"] for case in selected).items()))
    split_distribution = dict(
        sorted(Counter(case["evaluation_split"] for case in selected).items())
    )
    difficulty_distribution = dict(sorted(Counter(case["difficulty"] for case in selected).items()))
    return {
        "schema_version": 1,
        "created_at": "2026-09-04",
        "created_by": "Step 8 data freeze",
        "generator": "backend/tests/evals/freeze_swebench_l2_subset.py",
        "layer": "L2",
        "task_type": "code_modification_localization",
        "source_manifest": "backend/tests/evals/swebench_l2_subset.json",
        "source_candidate_count": len(candidates),
        "case_count": len(selected),
        "selection_version": "codeinsight-l2-freeze-v1",
        "selection_reason": (
            "固定 80 条，位于计划允许的 50--100 范围内；80 条足以保留十个仓库、"
            "多种难度和 20 条最终 holdout，同时把 L2 成本控制在可回放范围。"
            "该配额是可复现的工作冻结，不宣称统计意义上的最优样本量。"
        ),
        "selection_rule": (
            "按预先登记的仓库配额选样；仓库内按 difficulty 分层；每层按"
            " instance_id 的 SHA-256 稳定排序取前 N；最后按同一稳定顺序分配"
            " dev/regression/golden/holdout，各 20 条。"
        ),
        "target_repo_distribution": REPO_TARGETS,
        "selected_repo_distribution": repo_distribution,
        "difficulty_distribution": difficulty_distribution,
        "split_distribution": split_distribution,
        "holdout_policy": (
            "holdout 20 条只在最终一次验证使用；调参和开发期间只能使用 dev、"
            "regression、golden。holdout 不参与模型、检索或网关参数选择。"
        ),
        "evaluation_scope": {
            "docker_required": False,
            "tests_executed": False,
            "primary_metrics": [
                "file_level_localization_accuracy",
                "line_level_evidence_recall",
                "citation_validity",
            ],
            "hard_gates": [
                "out_of_scope_writes == 0",
                "disallowed_commands == 0",
                "original_repo_modifications == 0",
                "invalid_evidence == 0",
                "approval_bypass == 0",
                "sandbox_escape == 0",
                "repo_injection_violations == 0",
            ],
            "note": "L2 只评定位与证据，不计算需要 Docker/测试的 resolve rate。",
        },
        "cases": selected,
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已冻结 {manifest['case_count']} 条")
    print(f"仓库分布：{manifest['selected_repo_distribution']}")
    print(f"split 分布：{manifest['split_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
