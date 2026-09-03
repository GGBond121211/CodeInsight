"""runner 契约校验的回归测试。

运行方式（在仓库根目录）::

    uv run --project backend pytest experiments/ -q

这些测试保护的是 Step 1 的核心承诺：**不合规的实验必须被拒绝**。
如果某天有人放松了这些校验，实验结果的可信度就没有了。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runner import (
    REQUIRED_HARD_GATES,
    dataset_fingerprint,
    validate_file,
    validate_payload,
)

SPEC_PATH = Path("EXP-TEST.json")


def _valid_payload() -> dict:
    """一份完全合规的最小规格，各测试在它基础上做单点破坏。"""
    hard_gates: list[str] = []
    for name in REQUIRED_HARD_GATES:
        hard_gates.append(name)
    return {
        "experimentId": "EXP-001",
        "question": "chunk 行数与 overlap 如何影响证据召回？",
        "hypothesis": "80 行 0 overlap 不是最优，增加 overlap 可提高完整覆盖率。",
        "datasetVersion": "master_200",
        "split": "regression",
        "datasetHash": "0" * 64,
        "baselineProfile": "baseline-1.0",
        "candidateProfiles": ["chunk-40", "chunk-120"],
        "changedVariables": {"chunk_max_lines": [40, 80, 120]},
        "fixedVariables": {"rrf_k": 60, "semantic_min_score": 0.20},
        "metrics": ["evidence_recall", "complete_evidence_coverage"],
        "hardGates": hard_gates,
        "tier": "deterministic",
        "trials": 1,
        "status": "pending",
        "resultPath": "experiments/results/EXP-001/",
        "replayCommand": "python experiments/runner.py validate ...",
    }


def _issue_fields(report) -> set[str]:
    found: set[str] = set()
    for issue in report.issues:
        found.add(issue.field_name)
    return found


def test_valid_spec_is_accepted() -> None:
    report = validate_payload(_valid_payload(), SPEC_PATH)
    assert report.accepted, [issue.message for issue in report.issues]


def test_missing_baseline_is_refused() -> None:
    """没有基线一律拒绝——runner 存在的首要理由。"""
    payload = _valid_payload()
    payload["baselineProfile"] = None
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "baselineProfile" in _issue_fields(report)


def test_baseline_key_removed_is_refused() -> None:
    payload = _valid_payload()
    del payload["baselineProfile"]
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "baselineProfile" in _issue_fields(report)


def test_multiple_changed_variables_is_refused() -> None:
    """一次改两个变量，结果无法归因。"""
    payload = _valid_payload()
    payload["changedVariables"] = {
        "chunk_max_lines": [40, 80],
        "semantic_min_score": [0.0, 0.20],
    }
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "changedVariables" in _issue_fields(report)


def test_no_changed_variable_is_refused() -> None:
    payload = _valid_payload()
    payload["changedVariables"] = {}
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted


def test_missing_hard_gate_is_refused() -> None:
    """漏掉任何一条硬门槛都要拒绝，防止安全边界被静默绕过。"""
    payload = _valid_payload()
    payload["hardGates"] = ["invalid_evidence"]
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "hardGates" in _issue_fields(report)


def test_repo_injection_gate_is_required() -> None:
    """增补 R-2 新增的第 7 条门槛必须在必检列表里。"""
    assert "repo_injection_violations" in REQUIRED_HARD_GATES
    payload = _valid_payload()
    gates: list[str] = []
    for name in REQUIRED_HARD_GATES:
        if name != "repo_injection_violations":
            gates.append(name)
    payload["hardGates"] = gates
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted


@pytest.mark.parametrize(
    ("tier", "trials", "accepted"),
    [
        ("deterministic", 1, True),
        ("single_model_call", 1, False),
        ("single_model_call", 3, True),
        ("autonomous_tool_loop", 3, False),
        ("autonomous_tool_loop", 5, True),
    ],
)
def test_trial_minimum_by_tier(tier: str, trials: int, accepted: bool) -> None:
    """Trial 下限按变异来源分层。来源：DEC-0033。"""
    payload = _valid_payload()
    payload["tier"] = tier
    payload["trials"] = trials
    report = validate_payload(payload, SPEC_PATH)
    assert report.accepted is accepted


def test_unknown_tier_is_refused() -> None:
    payload = _valid_payload()
    payload["tier"] = "handwave"
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "tier" in _issue_fields(report)


def test_pending_spec_cannot_carry_verdict() -> None:
    """尚未运行就写结论，等于把计划当成事实。"""
    payload = _valid_payload()
    payload["verdict"] = "PROVEN"
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "verdict" in _issue_fields(report)


def test_completed_spec_may_carry_verdict() -> None:
    payload = _valid_payload()
    payload["status"] = "completed"
    payload["verdict"] = "DIRECTIONAL"
    report = validate_payload(payload, SPEC_PATH)
    assert report.accepted, [issue.message for issue in report.issues]


def test_unsupported_verdict_is_refused() -> None:
    payload = _valid_payload()
    payload["status"] = "completed"
    payload["verdict"] = "LOOKS_GOOD"
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted


def test_missing_required_field_is_refused() -> None:
    payload = _valid_payload()
    del payload["hypothesis"]
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "hypothesis" in _issue_fields(report)


def test_yaml_spec_is_refused_with_explanation(tmp_path: Path) -> None:
    """当前不引入 YAML 依赖，但要给出清楚的原因而不是崩溃。"""
    spec = tmp_path / "EXP-999.yaml"
    spec.write_text("experimentId: EXP-999\n", encoding="utf-8")
    report = validate_file(spec)
    assert not report.accepted
    assert "JSON" in report.issues[0].message


def test_broken_json_is_refused(tmp_path: Path) -> None:
    spec = tmp_path / "EXP-998.json"
    spec.write_text("{not json", encoding="utf-8")
    report = validate_file(spec)
    assert not report.accepted


def test_valid_file_roundtrip(tmp_path: Path) -> None:
    spec = tmp_path / "EXP-001.json"
    spec.write_text(json.dumps(_valid_payload(), ensure_ascii=False), encoding="utf-8")
    report = validate_file(spec)
    assert report.accepted, [issue.message for issue in report.issues]


def test_dataset_fingerprint_is_stable(tmp_path: Path) -> None:
    """数据集指纹用于检测静默改动，必须稳定且对内容敏感。"""
    dataset = tmp_path / "cases.json"
    dataset.write_text('{"cases": []}', encoding="utf-8")
    first = dataset_fingerprint(dataset)
    second = dataset_fingerprint(dataset)
    assert first == second
    assert len(first) == 64

    dataset.write_text('{"cases": [1]}', encoding="utf-8")
    assert dataset_fingerprint(dataset) != first


# ---------------------------------------------------------------------------
# 题型隔离（DEC-0038）
# ---------------------------------------------------------------------------


def test_mixing_task_types_is_refused() -> None:
    """只读问答与代码修改不得合并计算，否则会稀释掉 2.0 的核心能力。"""
    payload = _valid_payload()
    payload["datasetVersion"] = "change_tasks_l1"
    payload["datasets"] = ["master_200", "swebench_l2"]
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "datasets" in _issue_fields(report)


def test_same_task_type_datasets_are_accepted() -> None:
    payload = _valid_payload()
    payload["datasetVersion"] = "change_tasks_l1"
    payload["datasets"] = ["change_tasks_l1", "swebench_l2"]
    payload["taskType"] = "code_modification"
    report = validate_payload(payload, SPEC_PATH)
    assert report.accepted, [issue.message for issue in report.issues]


def test_unregistered_dataset_is_refused() -> None:
    """未登记的数据集会绕过题型检查，必须拒绝。"""
    payload = _valid_payload()
    payload["datasetVersion"] = "some_new_set"
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "datasets" in _issue_fields(report)


def test_declared_task_type_must_match_datasets() -> None:
    payload = _valid_payload()
    payload["datasetVersion"] = "master_200"
    payload["taskType"] = "code_modification"
    report = validate_payload(payload, SPEC_PATH)
    assert not report.accepted
    assert "taskType" in _issue_fields(report)


def test_dataset_version_may_carry_suffix() -> None:
    """允许 name@version 形式，按 @ 前的名称查表。"""
    payload = _valid_payload()
    payload["datasetVersion"] = "master_200@v2"
    payload["taskType"] = "comprehension"
    report = validate_payload(payload, SPEC_PATH)
    assert report.accepted, [issue.message for issue in report.issues]
