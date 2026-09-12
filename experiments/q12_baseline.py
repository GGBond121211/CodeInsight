"""Q-012 U0：先用事实测基线，再决定要不要做问题原型快路径。

为什么先测：快路径（U2）的收益全部建立在「简单高频问题本来会被开放 Tool Loop
花掉很多轮」这个假设上。如果 explain 的中位数只有一两次工具调用，那快路径
省下的不是成本，只是把同一件事换个地方做。

口径（只读，不写库）：
  - 每一行 = 一个 run（一次对话轮次）
  - task_type 取 agent_run_outputs.task_type
  - 工具调用次数 / 首个工具 / 模型轮数 / token 取自 run_events 与 result_json
  - P95 用最近邻法（nearest-rank），样本少时它就是最大值，不做插值假装样本充足

用法：
    backend/.venv/Scripts/python.exe experiments/q12_baseline.py
    backend/.venv/Scripts/python.exe experiments/q12_baseline.py --write-doc
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.application.evidence_ledger import EVIDENCE_TOOLS  # noqa: E402
from codeinsight.infrastructure.db.engine import (  # noqa: E402
    MySqlConfig,
    create_db_engine,
)
from codeinsight.infrastructure.model_gateway import (  # noqa: E402
    configured_change_token_budget,
    configured_explain_token_budget,
)
from sqlalchemy import text as sql_text  # noqa: E402

MODEL_RESULT = "model_result"
TOOL_COMMITTED = "tool_result_committed"
TOOL_REQUESTED = "tool_call_requested"

# A3 人工标注：键是真实取样里出现过的问题原文，值是（首个工具判断，原型候选名）。
# 只标注真正跑过的题；没标注的题在文档里照实写「未标注」，不用推断补满。
A3_ANNOTATIONS: dict[str, tuple[str, str]] = {
    "send 方法在哪里定义？": ("绕路：先要地图，再检索两次", "definition_lookup"),
    "Client 类定义在哪个文件？": ("绕路：先要地图，再检索两次", "definition_lookup"),
    "这个仓库的顶层模块有哪些？": ("合理：清点模块本来就该先要地图", "module_inventory"),
    "哪里实现了重定向跟随？": ("合理：检索优先，方向对", "definition_lookup"),
    "read_timeout 配置从哪里读取？": (
        "绕路：get_evidence_context 需要先有证据，首轮几乎拿不到",
        "config_source",
    ),
    "哪些模块引用了 Request 类？": (
        "绕路：先要地图，没有直接走引用查询",
        "call_sites",
    ),
}


def _load_env() -> None:
    """把仓库根的 .env 读进环境变量；不覆盖已有的值。"""

    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip()


@dataclass
class RunFact:
    run_id: str
    task_type: str
    assistant_chars: int = 0
    result: dict[str, object] = field(default_factory=dict)
    tool_calls: int = 0
    first_tool: str = ""
    model_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started_ms: int = 0
    finished_ms: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_ms(self) -> int:
        if not self.started_ms or not self.finished_ms:
            return 0
        return max(0, self.finished_ms - self.started_ms)


def _percentile(values: list[int], ratio: float) -> int:
    """最近邻分位数：样本少时宁可直接等于最大值，也不插值造精度。"""

    if not values:
        return 0
    ordered = sorted(values)
    if ratio <= 0:
        return ordered[0]
    import math

    rank = max(1, math.ceil(ratio * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _collect(engine) -> list[RunFact]:
    facts: dict[str, RunFact] = {}
    with engine.connect() as connection:
        for row in connection.execute(
            sql_text(
                "SELECT run_id, task_type, assistant_message, result_json "
                "FROM agent_run_outputs"
            )
        ):
            result: dict[str, object] = {}
            if row.result_json:
                try:
                    parsed = json.loads(row.result_json)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    result = parsed
            facts[row.run_id] = RunFact(
                run_id=row.run_id,
                task_type=row.task_type or "unknown",
                assistant_chars=len(row.assistant_message or ""),
                result=result,
            )
        for row in connection.execute(
            sql_text(
                "SELECT run_id, event_type, sequence, occurred_at_epoch_ms, payload_json "
                "FROM run_events ORDER BY run_id, sequence"
            )
        ):
            fact = facts.get(row.run_id)
            if fact is None:
                continue
            payload: dict[str, object] = {}
            if row.payload_json:
                try:
                    decoded = json.loads(row.payload_json)
                except ValueError:
                    decoded = None
                if isinstance(decoded, dict):
                    payload = decoded
            if not fact.started_ms:
                fact.started_ms = int(row.occurred_at_epoch_ms)
            fact.finished_ms = int(row.occurred_at_epoch_ms)
            if row.event_type == TOOL_COMMITTED:
                fact.tool_calls += 1
                if not fact.first_tool:
                    fact.first_tool = str(payload.get("tool_name", ""))
            elif row.event_type == TOOL_REQUESTED:
                # 只在没有 committed 事件时兜底：committed 才是「真的执行过」。
                if not fact.first_tool:
                    fact.first_tool = str(payload.get("tool_name", ""))
            elif row.event_type == MODEL_RESULT:
                fact.model_rounds += 1
                fact.input_tokens += int(str(payload.get("input_tokens", "0")) or 0)
                fact.output_tokens += int(str(payload.get("output_tokens", "0")) or 0)
    return list(facts.values())


def _summary(rows: list[RunFact], task_type: str) -> dict[str, object]:
    subset = [row for row in rows if row.task_type == task_type]
    tool_calls = [row.tool_calls for row in subset]
    tokens = [row.tokens for row in subset]
    return {
        "task_type": task_type,
        "runs": len(subset),
        "tool_calls_p50": int(statistics.median(tool_calls)) if tool_calls else 0,
        "tool_calls_p95": _percentile(tool_calls, 0.95),
        "tool_calls_max": max(tool_calls) if tool_calls else 0,
        "tokens_p50": int(statistics.median(tokens)) if tokens else 0,
        "tokens_p95": _percentile(tokens, 0.95),
        "tokens_max": max(tokens) if tokens else 0,
        "with_zero_tools": sum(1 for value in tool_calls if value == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q-012 U0 基线统计")
    parser.add_argument("--write-doc", action="store_true", help="写入 docs/Q12_BASELINE.md")
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=ROOT / "work" / "q12_explain_probe.json",
        help="真实模型 explain 取样的 JSON（q12_explain_probe.py 产出）",
    )
    args = parser.parse_args()

    _load_env()
    config = MySqlConfig.from_env()
    if config is None:
        print("未配置 MySQL（CODEINSIGHT_MYSQL_HOST/USER/DATABASE），无法统计。")
        return 2
    engine = create_db_engine(config)
    rows = _collect(engine)
    if not rows:
        print("库里没有任何 agent_run_outputs 记录：没有可统计的事实。")
        return 3

    per_type = Counter(row.task_type for row in rows)
    summaries = [_summary(rows, task_type) for task_type in sorted(per_type)]
    explain = [row for row in rows if row.task_type == "explain"]
    first_tools = Counter(row.first_tool for row in explain if row.first_tool)

    print(f"run 总数：{len(rows)}；按类型：{dict(per_type)}")
    for item in summaries:
        print(json.dumps(item, ensure_ascii=False))
    print("explain 首个工具分布：", dict(first_tools))
    empty_rounds = [
        row for row in explain if not row.first_tool
    ]
    print(f"explain 中没有任何工具调用的 run：{len(empty_rounds)}")

    probe = _load_probe(args.probe_json)
    if probe is not None:
        print("真实取样结果：", json.dumps(probe, ensure_ascii=False))

    if args.write_doc:
        path = ROOT / "docs" / "Q12_BASELINE.md"
        path.write_text(
            _render(rows, summaries, first_tools, probe),
            encoding="utf-8",
            newline="\n",
        )
        print(f"已写入 {path}")
    return 0


def _load_probe(path: Path) -> dict[str, object] | None:
    """读取真实取样；文件不存在时返回 None，而不是伪造一份样本。"""

    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    records = parsed.get("records")
    if not isinstance(records, list) or not records:
        return None
    tool_calls = [int(item.get("tool_calls", 0) or 0) for item in records]
    tokens = [
        int(item.get("input_tokens", 0) or 0) + int(item.get("output_tokens", 0) or 0)
        for item in records
    ]
    first_tools = Counter(
        str(item.get("first_tool", "")) for item in records if item.get("first_tool")
    )
    statuses = Counter(str(item.get("status", "")) for item in records)
    return {
        "records": len(records),
        "model": parsed.get("model", ""),
        "tool_calls_p50": int(statistics.median(tool_calls)),
        "tool_calls_p95": _percentile(tool_calls, 0.95),
        "tool_calls_max": max(tool_calls),
        "tokens_p50": int(statistics.median(tokens)),
        "tokens_p95": _percentile(tokens, 0.95),
        "tokens_max": max(tokens),
        "total_tokens": int(parsed.get("total_tokens", 0) or 0),
        "first_tools": dict(first_tools),
        "statuses": dict(statuses),
        "items": [
            {
                "question": str(item.get("question", "")),
                "first_tool": str(item.get("first_tool", "")),
                "tool_calls": int(item.get("tool_calls", 0) or 0),
                "status": str(item.get("status", "")),
            }
            for item in records
        ],
    }


def _render_annotations(items: list[dict[str, object]]) -> list[str]:
    """A3：逐题标注首个工具是否合理，并把原型候选列出来。"""

    if not items:
        return []
    lines = [
        "## A3 标注：首个工具是否合理、是否属于高频原型",
        "",
        "| 问题 | 首个工具 | 工具调用 | 首个工具判断 | 原型候选 |",
        "| --- | --- | --- | --- | --- |",
    ]
    candidates: Counter[str] = Counter()
    evidence_first = 0
    evidence_tools_label = "、".join(sorted(EVIDENCE_TOOLS))
    for item in items:
        question = str(item.get("question", ""))
        first_tool = str(item.get("first_tool", ""))
        if first_tool in EVIDENCE_TOOLS:
            evidence_first += 1
        annotation = A3_ANNOTATIONS.get(question)
        judgement = annotation[0] if annotation else "未标注"
        archetype = annotation[1] if annotation else "未标注"
        if annotation:
            candidates[archetype] += 1
        calls = item.get("tool_calls", 0)
        lines.append(
            f"| {question} | `{first_tool}` | {calls} | {judgement} | {archetype} |"
        )
    lines.extend(
        [
            "",
            f"首个工具就落在取证工具（{evidence_tools_label}）上的题目："
            f"{evidence_first}/{len(items)}。",
            "",
            "原型候选（按本次取样出现次数）：",
            "",
        ]
    )
    if candidates:
        for name, count in candidates.most_common():
            lines.append(f"- `{name}`：{count}")
    else:
        lines.append("- （本次取样没有命中任何已标注原型）")
    lines.extend(
        [
            "",
            "这里只是**候选**：候选名还没有独立配方与阈值，运行时也不生效",
            "（快路径默认关闭，属 U2 范围，要等 C2 的分阶段指标才允许打开）。",
            "首个工具选择与最终证据召回的关系，本次样本还不足以下结论，",
            "由 C2.3 单独测「首工具正确率」。",
            "",
        ]
    )
    return lines


def _render(
    rows: list[RunFact],
    summaries: list[dict[str, object]],
    first_tools: Counter,
    probe: dict[str, object] | None,
) -> str:
    lines = [
        "# Q-012 U0：explain / change 的既有事实基线",
        "",
        "本文件由 `experiments/q12_baseline.py --write-doc` 生成，数据来自本机 MySQL 的",
        "`agent_run_outputs` 与 `run_events`；不写库、不改任何状态。",
        "",
        f"样本：{len(rows)} 个 run。",
        "",
        "## 分类型分布",
        "",
        "| task_type | runs | 工具调用 P50 | P95 | max | token P50 | P95 | max | 零工具 run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summaries:
        lines.append(
            "| {task_type} | {runs} | {tool_calls_p50} | {tool_calls_p95} | {tool_calls_max} "
            "| {tokens_p50} | {tokens_p95} | {tokens_max} | {with_zero_tools} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## explain 的首个工具分布",
            "",
        ]
    )
    if first_tools:
        for name, count in first_tools.most_common():
            lines.append(f"- `{name}`：{count}")
    else:
        lines.append("- （没有可统计的工具调用）")
    lines.append("")
    lines.extend(["## 真实模型取样（A2，小批量）", ""])
    if probe is None:
        lines.extend(["没有找到取样文件；本节留空而不是用估算数字填满。", ""])
        return "\n".join(lines)
    lines.extend(
        [
            f"样本：{probe['records']} 个单题 explain，模型 `{probe['model']}`，",
            "由 `experiments/q12_explain_probe.py` 对真实 Provider 跑出；",
            f"合计 {probe['total_tokens']} token。",
            "",
            "| 指标 | P50 | P95 | max |",
            "| --- | --- | --- | --- |",
            f"| 工具调用次数 | {probe['tool_calls_p50']} | {probe['tool_calls_p95']} "
            f"| {probe['tool_calls_max']} |",
            f"| 输入+输出 token | {probe['tokens_p50']} | {probe['tokens_p95']} "
            f"| {probe['tokens_max']} |",
            "",
            f"首个工具分布：{probe['first_tools']}",
            "",
            f"终止状态分布：{probe['statuses']}",
            "",
        ]
    )
    probe_items = probe.get("items")
    lines.extend(
        _render_annotations(list(probe_items) if isinstance(probe_items, list) else [])
    )
    # 闸值直接读代码里的默认值，避免文档写一套、实现另一套。
    explain_budget = configured_explain_token_budget()
    change_budget = configured_change_token_budget()
    lines.extend(
        [
            "## 决策闸结论",
            "",
        ]
    )
    if probe["tool_calls_p50"] <= 2 and probe["tokens_p95"] < probe["tokens_p50"] * 3:
        lines.extend(
            [
                "满足「工具调用中位数 ≤ 2 且 P95 token 不足 P50 的 3 倍」：",
                "**U2 问题原型快路径不做**，U6 只接线不做压缩分层（已按此实施）。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "不满足「中位数 ≤ 2 且 P95 token 不足 P50 的 3 倍」：",
                "**U2 做**，但默认关闭，必须等 C2 的分阶段指标出结果才允许打开。",
                "",
                "U6 的取值口径同时说明：本次取样的 token 量级只有万级，所以默认花费闸",
                f"（explain {explain_budget:,} / change {change_budget:,}）不是按 P95 卡出来的，",
                "而是留了一个数量级的余量专门拦失控循环；把闸设到 P95 会误伤正常",
                "的复杂问题。",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
