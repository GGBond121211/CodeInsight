"""change task 清单的契约测试。

与既有的 test_master_200_cases.py 等资产测试同一职责：保证数据集结构不被静默破坏。

这些断言保护的是 Step 1 的承诺——每个样本都有明确的边界、验收条件和来源说明。
如果某天有人放松了它们，L1 数据集就退化成一堆没有约束的自然语言描述。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("change_eval_manifest.json")
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

EXPECTED_COUNTS = {
    "single_file_fix": 8,
    "cross_file_change": 8,
    "failing_test_repair": 6,
    "insufficient_evidence_refusal": 4,
    "attack_sample": 4,
    "prompt_injection_in_repo": 4,
}

SUPPORTED_SPLITS = frozenset(
    {"smoke", "regression", "golden", "boundary", "security", "reliability", "holdout"}
)

SUPPORTED_AUTHORING_STATUS = frozenset(
    {"specified", "described_not_written", "needs_fixture", "needs_case_selection"}
)

# 七条硬门槛。来源：docs/BOUNDARIES.md。
KNOWN_HARD_GATES = frozenset(
    {
        "out_of_scope_writes",
        "disallowed_commands",
        "original_repo_modifications",
        "invalid_evidence",
        "approval_bypass",
        "sandbox_escape",
        "repo_injection_violations",
    }
)


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_exists_and_parses() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["generator"].endswith("build_change_eval_manifest.py")


def test_total_task_count() -> None:
    manifest = load_manifest()
    assert len(manifest["tasks"]) == 34


def test_category_distribution_matches_plan() -> None:
    """分布来自 PLAN_2.0.md Step 1 加上增补 R-2 的注入类。"""
    manifest = load_manifest()
    assert manifest["counts_by_category"] == EXPECTED_COUNTS


def test_task_ids_are_unique_and_sequential() -> None:
    manifest = load_manifest()
    ids: list[str] = []
    for task in manifest["tasks"]:
        ids.append(task["task_id"])
    assert len(set(ids)) == len(ids)
    expected: list[str] = []
    for number in range(1, 35):
        expected.append(f"CT-{number:03d}")
    assert ids == expected


def test_every_task_has_required_structure() -> None:
    manifest = load_manifest()
    required = [
        "task_id",
        "category",
        "split",
        "fixture",
        "user_goal",
        "target_files",
        "allowed_files",
        "forbidden_files",
        "validation_profile",
        "reference_patch",
        "should_refuse",
        "acceptance",
        "hard_gates_exercised",
        "authoring_status",
    ]
    for task in manifest["tasks"]:
        for field_name in required:
            assert field_name in task, f"{task['task_id']} 缺少 {field_name}"


def test_splits_are_supported() -> None:
    manifest = load_manifest()
    for task in manifest["tasks"]:
        assert task["split"] in SUPPORTED_SPLITS, task["task_id"]


def test_authoring_status_is_supported() -> None:
    manifest = load_manifest()
    for task in manifest["tasks"]:
        assert task["authoring_status"] in SUPPORTED_AUTHORING_STATUS, task["task_id"]
        legend = manifest["authoring_status_legend"]
        assert task["authoring_status"] in legend


def test_hard_gates_are_known() -> None:
    """不允许出现拼错或臆造的门槛名，否则实验里会静默漏检。"""
    manifest = load_manifest()
    for task in manifest["tasks"]:
        for gate in task["hard_gates_exercised"]:
            assert gate in KNOWN_HARD_GATES, f"{task['task_id']} 出现未知门槛 {gate}"


def test_every_task_checks_invalid_evidence() -> None:
    """无效 evidence 是所有任务的共同底线。"""
    manifest = load_manifest()
    for task in manifest["tasks"]:
        assert "invalid_evidence" in task["hard_gates_exercised"], task["task_id"]


def test_acceptance_is_non_empty() -> None:
    manifest = load_manifest()
    for task in manifest["tasks"]:
        assert len(task["acceptance"]) >= 2, f"{task['task_id']} 的验收条件过少"


def test_forbidden_files_always_present() -> None:
    """禁止路径必须逐条声明，不能因为某个任务看起来无害就省略。"""
    manifest = load_manifest()
    for task in manifest["tasks"]:
        forbidden = task["forbidden_files"]
        assert "**/.env" in forbidden, task["task_id"]
        assert ".git/**" in forbidden, task["task_id"]


def test_refusal_and_attack_tasks_expect_refusal() -> None:
    manifest = load_manifest()
    refusing = {"insufficient_evidence_refusal", "attack_sample"}
    for task in manifest["tasks"]:
        if task["category"] in refusing:
            assert task["should_refuse"] is True, task["task_id"]
            assert task["target_files"] == [], f"{task['task_id']} 拒答类不应有目标文件"


def test_repair_tasks_do_not_expect_refusal() -> None:
    manifest = load_manifest()
    repairing = {"single_file_fix", "cross_file_change", "failing_test_repair"}
    for task in manifest["tasks"]:
        if task["category"] in repairing:
            assert task["should_refuse"] is False, task["task_id"]


def test_injection_tasks_assert_both_directions() -> None:
    """增补 R-2 的核心要求：不能用「一律拒答」通过注入测试。"""
    manifest = load_manifest()
    found = 0
    for task in manifest["tasks"]:
        if task["category"] != "prompt_injection_in_repo":
            continue
        found += 1
        assert task["should_refuse"] is False, f"{task['task_id']} 注入类不应整体拒答"
        assert "repo_injection_violations" in task["hard_gates_exercised"]
        joined = " ".join(task["acceptance"])
        assert "不执行注入指令" in joined, task["task_id"]
        assert "仍能正确回答" in joined, f"{task['task_id']} 缺少正向能力断言"
    assert found == 4


def test_sample_repo_target_files_exist() -> None:
    """针对 sample_repo 的目标文件必须真实存在，防止清单指向不存在的路径。"""
    manifest = load_manifest()
    checked = 0
    for task in manifest["tasks"]:
        if task["fixture"]["repo"] != "sample_repo":
            continue
        for relative in task["target_files"]:
            candidate = FIXTURE_ROOT / relative
            assert candidate.is_file(), f"{task['task_id']} 指向不存在的文件 {relative}"
            checked += 1
    assert checked >= 16


def test_external_repo_tasks_pin_commit() -> None:
    """外部仓库任务必须固定 tag 与 commit，否则行号期望会随版本漂移。"""
    manifest = load_manifest()
    found = 0
    for task in manifest["tasks"]:
        if task["category"] != "failing_test_repair":
            continue
        found += 1
        fixture = task["fixture"]
        assert fixture["repo"] in {"httpx", "click", "requests"}
        assert len(fixture["commit"]) == 40, task["task_id"]
        assert fixture["tag"], task["task_id"]
        assert fixture["path"].startswith("work/benchmarks/")
    assert found == 6


def test_allowed_files_never_exceed_targets() -> None:
    """允许改动的范围不得大于目标范围，这是 diff 范围门禁的基础。"""
    manifest = load_manifest()
    for task in manifest["tasks"]:
        assert set(task["allowed_files"]) <= set(task["target_files"]), task["task_id"]


def test_statistical_note_is_present() -> None:
    """样本量约束必须写在清单里，防止后续把 34 案的方向性结论当成已证明。"""
    manifest = load_manifest()
    note = manifest["statistical_note"]
    assert "0.13" in note
    assert "DEC-0033" in note
