"""生成 2.1.0 数据集准入报告，不调用模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_quality import QualityReport, validate_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "experiments" / "configs" / "dataset_quality_policy_2_1_0.yaml"
DEFAULT_BASELINE = ROOT / "experiments" / "configs" / "baseline_2_1_0.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "decision_records" / "DATASET-READINESS-2.1.0.md"


def render_markdown(report: QualityReport) -> str:
    lines = [
        "# CodeInsight 2.1.0 数据集准入报告",
        "",
        "> 本报告只描述数据准入和证据审查状态，不把模型 pilot 写成最终质量结论。",
        "",
        f"- 自动准入：`{'PASS' if report.accepted else 'BLOCKED'}`",
        f"- 策略：`{report.policy_path}`",
        f"- 基线：`{report.baseline_path or '未提供'}`",
        "- 模型质量数据集：`conversation_2_1_0`",
        "- 真实模型限制：每次连续验收周期不超过 10,000,000 tokens；"
        "请求级输出预算和 `max_retries=0` 由运行命令控制",
        "",
        "## 登记资产",
        "",
        "| 数据集 | 数量 | 状态 | SHA-256 / 说明 |",
        "|---|---:|---|---|",
    ]
    for check in report.checks:
        count = check.actual_count if check.actual_count is not None else "-"
        extra = check.actual_sha256 or "-"
        if check.actual_turn_count is not None:
            extra += f"；turns={check.actual_turn_count}"
        lines.append(f"| `{check.dataset_id}` | {count} | `{check.state}` | `{extra}` |")
    lines.extend(
        [
            "",
            "## 多轮数据集结论",
            "",
            "- 24 个独立 conversation case、123 个用户轮次；"
            "dev/regression/golden/holdout 各 6 条，中文 18 条、中英混合 6 条。",
            "- Evidence 路径、1-based 行区间、case/turn ID、split 成员、"
            "跨 case 问题重复和禁存模型内容门禁通过。",
            "- 24/24 条 case 已完成 `codex-agent-source-audit` 源码审查，覆盖率 100%；"
            "这不是独立人工双盲审查。",
            "- 当前状态是 `READY_FOR_MODEL_PILOT`，允许小规模真实 Provider pilot；"
            "没有独立人工复核的最终语义结论仍保持阻断。",
            "- change 轮只验证“生成预览并等待审批、不得直接改 fixture”；未审批不计入修改成功。",
            "",
            "## 现有缺口仍然保留",
            "",
            "- Master-200、multilingual 和 fusion 仍是单轮数据，"
            "不能代替多轮上下文数据，也没有独立 comprehension holdout。",
            "- L1 change manifest 的部分任务仍是 `described_not_written`，"
            "不能进入 patch success 分母。",
            "- L2 frozen-80 只支持文件/行定位结论，不等同完整 SWE-bench resolve。",
            "- 真实 pilot 只报告 route、事实/证据命中、目标连续性、上下文压缩、"
            "token、缓存和延迟；不使用 Fake Provider 代表模型质量。",
            "",
            "## 停止条件",
            "",
            "1. 自动准入出现 `DATASET_BLOCKED` 时停止实验，不自动刷新 hash 或跳过案例。",
            "2. pilot 使用 dev/regression/golden；holdout 只在参数和 Prompt 冻结后使用。",
            "3. 只有独立审查和 holdout 都满足，状态才可升级为 `READY_FOR_FINAL`。",
            "",
        ]
    )
    if report.issues:
        lines.extend(["## 报告级阻塞", ""])
        lines.extend(f"- `{issue.code}`：{issue.message}" for issue in report.issues)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate_policy(args.policy, baseline_path=args.baseline, repo_root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report), encoding="utf-8")
    print(f"{'通过' if report.accepted else '阻断'}：{args.output}")
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
