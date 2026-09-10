"""实验 runner：加载 ExperimentSpec，校验契约，拒绝不合规的实验。

这个模块**不负责跑实验**，只负责在实验开始前把关。真正的实验实现随各 Step 落地。

为什么需要它：
    没有基线的数字无法解释。「Evidence Recall 0.86」是好是坏？脱离基线就不知道。
    允许无基线运行等于允许产生一堆无法归因的数字，而这些数字最终会被写进
    简历——那是最坏的结果。

四条硬性拒绝：
    1. 缺少必填字段            → 契约不完整，事后无法复现
    2. 没有 baselineProfile    → 无法解释结果
    3. 同时改动多个变量        → 无法归因
    4. 未声明 hardGates        → 安全门禁可能被静默绕过

用法::

    python experiments/runner.py validate experiments/specs/EXP-001.yaml
    python experiments/runner.py list experiments/specs/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# ExperimentSpec 的必填字段。来源：docs/PLAN_2.0.md Step 1「ExperimentSpec 最小结构」。
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "experimentId",
        "question",
        "hypothesis",
        "datasetVersion",
        "split",
        "baselineProfile",
        "candidateProfiles",
        "changedVariables",
        "fixedVariables",
        "metrics",
        "hardGates",
        "trials",
        "resultPath",
    }
)

# 数据集 -> 题型 注册表。来源：DECISIONS.md DEC-0038。
#
# 两类题测的根本不是同一个东西，合并平均会让「很会找代码但改不对代码」的系统
# 拿到虚高的总分。未登记的数据集一律拒绝，防止新数据集绕过分类。
DATASET_TASK_TYPES: dict[str, str] = {
    # 题型 A：只读问答。角色是回归护栏，只看有没有下降，绝不进 2.0 头条数字。
    "master_200": "comprehension",
    "fusion_131": "comprehension",
    "temporary_30": "comprehension",
    "multilingual_v1": "comprehension",
    "multilingual_v2": "comprehension",
    "httpx_13": "comprehension",
    "httpx_answer_8": "comprehension",
    "answer_13": "comprehension",
    "retrieval_27": "comprehension",
    # 题型 C：同一 Session 的多轮体验。它与单轮 comprehension 分开报告，
    # 因为它同时测量入口切换、目标延续、上下文保留和范围引导。
    "conversation_2_1_0": "conversation_quality",
    # 题型 B：代码修改。2.0 能力主指标的唯一来源。
    "change_tasks_l1": "code_modification",
    "swebench_l2": "code_modification",
    "swebench_l3": "code_modification",
    "swebench_l4": "code_modification",
}

SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(DATASET_TASK_TYPES.values())

# 允许的实验状态。pending 表示已登记但尚未运行——禁止把 pending 当作已有结论。
SUPPORTED_STATUSES: frozenset[str] = frozenset({"pending", "running", "completed", "abandoned"})

# 允许的结论分级。来源：DECISIONS.md DEC-0033。
SUPPORTED_VERDICTS: frozenset[str] = frozenset(
    {"PROVEN", "DIRECTIONAL", "NO_EFFECT", "REGRESSION"}
)

# 按变异来源分层的 Trial 数下限。来源：DEC-0033。
MIN_TRIALS_BY_TIER: dict[str, int] = {
    "deterministic": 1,
    "single_model_call": 3,
    "autonomous_tool_loop": 5,
}

# 七条硬门槛。来源：docs/BOUNDARIES.md。任何实验都必须声明它检查了这些。
REQUIRED_HARD_GATES: tuple[str, ...] = (
    "out_of_scope_writes",
    "disallowed_commands",
    "original_repo_modifications",
    "invalid_evidence",
    "approval_bypass",
    "sandbox_escape",
    "repo_injection_violations",
)


@dataclass(frozen=True)
class ValidationIssue:
    """一条校验问题。``fatal`` 为真时拒绝运行。"""

    field_name: str
    message: str
    fatal: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    """已经过基本解析的实验规格。字段语义见 docs/PLAN_2.0.md。"""

    raw: dict[str, Any]
    path: Path

    @property
    def experiment_id(self) -> str:
        value = self.raw.get("experimentId")
        if isinstance(value, str):
            return value
        return "(未命名)"

    @property
    def tier(self) -> str:
        value = self.raw.get("tier")
        if isinstance(value, str):
            return value
        return "unknown"

    @property
    def status(self) -> str:
        value = self.raw.get("status")
        if isinstance(value, str):
            return value
        return "pending"


@dataclass
class ValidationReport:
    """一次校验的完整结果。"""

    spec_path: Path
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def fatal_issues(self) -> list[ValidationIssue]:
        found: list[ValidationIssue] = []
        for issue in self.issues:
            if issue.fatal:
                found.append(issue)
        return found

    @property
    def accepted(self) -> bool:
        return len(self.fatal_issues) == 0


def _load_mapping(path: Path) -> dict[str, Any]:
    """读取 YAML 或 JSON 规格文件。

    刻意不引入 PyYAML：Step 1 不为「以后可能需要」安装依赖。
    YAML 规格由调用方先转成 JSON，或直接写 JSON。
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            f"{path.name}：当前 runner 只解析 JSON。"
            "YAML 规格请先转换，或等 Step 8 明确需要时再引入 YAML 依赖。"
        )
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}：ExperimentSpec 必须是一个 JSON 对象")
    return payload


def dataset_fingerprint(path: Path) -> str:
    """计算数据集内容指纹，用于检测数据集被静默改动。"""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _check_required_fields(payload: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in sorted(REQUIRED_FIELDS):
        if name not in payload:
            issues.append(
                ValidationIssue(
                    field_name=name,
                    message="缺少必填字段。契约不完整，事后无法复现。",
                )
            )
    return issues


def _check_baseline(payload: dict[str, Any]) -> list[ValidationIssue]:
    """没有基线一律拒绝——这是 runner 存在的首要理由。"""
    issues: list[ValidationIssue] = []
    baseline = payload.get("baselineProfile")
    if not baseline:
        issues.append(
            ValidationIssue(
                field_name="baselineProfile",
                message=(
                    "未声明基线。脱离基线的指标无法解释，"
                    "也无法回答「相比什么提升了」。拒绝运行。"
                ),
            )
        )
        return issues
    if not isinstance(baseline, str | dict):
        issues.append(
            ValidationIssue(
                field_name="baselineProfile",
                message="基线必须是 profile 名称或 profile 对象。",
            )
        )
    return issues


def _check_single_variable(payload: dict[str, Any]) -> list[ValidationIssue]:
    """强制单变量。一次改两个变量，结果无法归因到任何一个。"""
    issues: list[ValidationIssue] = []
    changed = payload.get("changedVariables")
    if changed is None:
        return issues
    if not isinstance(changed, dict):
        issues.append(
            ValidationIssue(
                field_name="changedVariables",
                message="changedVariables 必须是「变量名 → 候选值列表」的对象。",
            )
        )
        return issues
    if len(changed) > 1:
        names = sorted(changed)
        issues.append(
            ValidationIssue(
                field_name="changedVariables",
                message=(
                    f"同时改动了 {len(changed)} 个变量：{names}。"
                    "结果无法归因到任何一个。请拆成多个实验。"
                ),
            )
        )
    if len(changed) == 0:
        issues.append(
            ValidationIssue(
                field_name="changedVariables",
                message="没有声明任何被改动的变量，这不构成一次对照实验。",
            )
        )
    return issues


def _check_hard_gates(payload: dict[str, Any]) -> list[ValidationIssue]:
    """硬门槛必须显式声明，防止优化过程静默侵蚀安全边界。"""
    issues: list[ValidationIssue] = []
    gates = payload.get("hardGates")
    if gates is None:
        return issues
    if not isinstance(gates, list):
        issues.append(
            ValidationIssue(
                field_name="hardGates",
                message="hardGates 必须是列表。",
            )
        )
        return issues
    declared: set[str] = set()
    for item in gates:
        if isinstance(item, str):
            declared.add(item)
    missing: list[str] = []
    for name in REQUIRED_HARD_GATES:
        if name not in declared:
            missing.append(name)
    if missing:
        issues.append(
            ValidationIssue(
                field_name="hardGates",
                message=(
                    f"未声明这些硬门槛：{missing}。"
                    "任一门槛不为 0 时该候选应直接淘汰，不看质量分。"
                ),
            )
        )
    return issues


def _check_trials(payload: dict[str, Any]) -> list[ValidationIssue]:
    """按实验分层校验 Trial 数是否达到下限。来源：DEC-0033。"""
    issues: list[ValidationIssue] = []
    trials = payload.get("trials")
    tier = payload.get("tier")
    if not isinstance(trials, int):
        if trials is not None:
            issues.append(
                ValidationIssue(field_name="trials", message="trials 必须是整数。")
            )
        return issues
    if trials <= 0:
        issues.append(ValidationIssue(field_name="trials", message="trials 必须为正整数。"))
        return issues
    if not isinstance(tier, str) or tier not in MIN_TRIALS_BY_TIER:
        issues.append(
            ValidationIssue(
                field_name="tier",
                message=(
                    f"tier 必须是 {sorted(MIN_TRIALS_BY_TIER)} 之一，"
                    "否则无法判断 Trial 数是否足够。"
                ),
            )
        )
        return issues
    minimum = MIN_TRIALS_BY_TIER[tier]
    if trials < minimum:
        issues.append(
            ValidationIssue(
                field_name="trials",
                message=(
                    f"tier={tier} 要求至少 {minimum} 次 Trial，当前为 {trials}。"
                    "Trial 不足时点估计无法与采样波动区分。"
                ),
            )
        )
    return issues


def _check_status_and_verdict(payload: dict[str, Any]) -> list[ValidationIssue]:
    """状态与结论分级必须合法；pending 不得携带结论。"""
    issues: list[ValidationIssue] = []
    status = payload.get("status", "pending")
    if isinstance(status, str) and status not in SUPPORTED_STATUSES:
        issues.append(
            ValidationIssue(
                field_name="status",
                message=f"status 必须是 {sorted(SUPPORTED_STATUSES)} 之一。",
            )
        )
    verdict = payload.get("verdict")
    if verdict is None:
        return issues
    if not isinstance(verdict, str) or verdict not in SUPPORTED_VERDICTS:
        issues.append(
            ValidationIssue(
                field_name="verdict",
                message=f"verdict 必须是 {sorted(SUPPORTED_VERDICTS)} 之一。",
            )
        )
        return issues
    if status == "pending":
        issues.append(
            ValidationIssue(
                field_name="verdict",
                message=(
                    "status=pending 的实验不得携带 verdict。"
                    "尚未运行就写结论，是把计划当成事实。"
                ),
            )
        )
    return issues


def _check_task_type_isolation(payload: dict[str, Any]) -> list[ValidationIssue]:
    """禁止一次实验同时引用只读问答与代码修改两类数据集。

    来源：DEC-0038。这两类题测的能力不同，混在一起算平均分会稀释掉
    2.0 真正想证明的代码修改能力。
    """
    issues: list[ValidationIssue] = []
    names: list[str] = []
    raw = payload.get("datasets")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                names.append(item)
    single = payload.get("datasetVersion")
    if isinstance(single, str) and single:
        names.append(single)
    if not names:
        return issues

    unknown: list[str] = []
    observed: set[str] = set()
    for name in names:
        base = name.split("@", 1)[0]
        task_type = DATASET_TASK_TYPES.get(base)
        if task_type is None:
            unknown.append(name)
        else:
            observed.add(task_type)
    if unknown:
        issues.append(
            ValidationIssue(
                field_name="datasets",
                message=(
                    f"这些数据集未在 DATASET_TASK_TYPES 中登记：{unknown}。"
                    "未登记的数据集会绕过题型隔离检查，请先登记其题型。"
                ),
            )
        )
    if len(observed) > 1:
        issues.append(
            ValidationIssue(
                field_name="datasets",
                message=(
                    f"同一实验混用了多种题型：{sorted(observed)}。"
                    "只读问答是回归护栏、代码修改是能力主指标，两者不得合并计算（DEC-0038）。"
                ),
            )
        )
    declared = payload.get("taskType")
    if isinstance(declared, str) and observed and declared not in observed:
        issues.append(
            ValidationIssue(
                field_name="taskType",
                message=(
                    f"声明的 taskType={declared} "
                    f"与所引用数据集的题型 {sorted(observed)} 不符。"
                ),
            )
        )
    return issues


def validate_payload(payload: dict[str, Any], spec_path: Path) -> ValidationReport:
    """对一份已解析的规格执行全部校验。"""
    report = ValidationReport(spec_path=spec_path)
    report.issues.extend(_check_required_fields(payload))
    report.issues.extend(_check_baseline(payload))
    report.issues.extend(_check_single_variable(payload))
    report.issues.extend(_check_hard_gates(payload))
    report.issues.extend(_check_trials(payload))
    report.issues.extend(_check_status_and_verdict(payload))
    report.issues.extend(_check_task_type_isolation(payload))
    return report


def validate_file(path: Path) -> ValidationReport:
    """读取并校验一个规格文件。解析失败本身也是一条致命问题。"""
    try:
        payload = _load_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = ValidationReport(spec_path=path)
        report.issues.append(ValidationIssue(field_name="(file)", message=str(error)))
        return report
    return validate_payload(payload, path)


def _print_report(report: ValidationReport) -> None:
    name = report.spec_path.name
    if report.accepted:
        print(f"[通过] {name}")
        for issue in report.issues:
            print(f"    提醒 · {issue.field_name}：{issue.message}")
        return
    print(f"[拒绝] {name}")
    for issue in report.issues:
        marker = "致命" if issue.fatal else "提醒"
        print(f"    {marker} · {issue.field_name}：{issue.message}")


def _command_validate(paths: list[Path]) -> int:
    exit_code = 0
    for path in paths:
        report = validate_file(path)
        _print_report(report)
        if not report.accepted:
            exit_code = 1
    return exit_code


def _command_list(directory: Path) -> int:
    if not directory.is_dir():
        print(f"不是目录：{directory}", file=sys.stderr)
        return 2
    spec_paths = sorted(directory.glob("*.json"))
    if not spec_paths:
        print(f"{directory} 下没有 .json 规格文件。")
        return 0
    print(f"{'实验 ID':<12} {'分层':<22} {'状态':<12} 文件")
    for path in spec_paths:
        try:
            payload = _load_mapping(path)
        except (OSError, ValueError, json.JSONDecodeError):
            print(f"{'(解析失败)':<12} {'-':<22} {'-':<12} {path.name}")
            continue
        spec = ExperimentSpec(raw=payload, path=path)
        print(f"{spec.experiment_id:<12} {spec.tier:<22} {spec.status:<12} {path.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiments/runner.py",
        description="ExperimentSpec 契约校验器。不跑实验，只在实验开始前把关。",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="校验一个或多个规格文件")
    validate.add_argument("paths", nargs="+", metavar="SPEC")

    listing = commands.add_parser("list", help="列出目录下的全部规格及其状态")
    listing.add_argument("directory", metavar="DIR")

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "validate":
        paths: list[Path] = []
        for item in arguments.paths:
            paths.append(Path(item))
        return _command_validate(paths)
    if arguments.command == "list":
        return _command_list(Path(arguments.directory))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
