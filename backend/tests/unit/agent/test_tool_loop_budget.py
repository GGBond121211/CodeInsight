"""Q-012 花费闸的契约测试。

三件事必须成立：超预算停在本地而不抛异常、70% 软阈值只触发一次、
没有声明预算时不假装知道「用掉了几成」。
"""

from __future__ import annotations

import pytest

from codeinsight.agent.tool_loop import (
    ToolCall,
    ToolLoop,
    ToolLoopConfig,
    ToolModelResponse,
    ToolResult,
)
from codeinsight.domain.trace import BUDGET_EXHAUSTED, BUDGET_LIMIT, BUDGET_USED
from codeinsight.infrastructure.event_log import InMemoryEventLog
from codeinsight.infrastructure.tool_registry import build_default_registry


class Host:
    def __init__(self) -> None:
        self.tools = build_default_registry().list_tools()

    def list_tools(self):
        return self.tools

    def call_tool(self, call):
        return ToolResult.success(call.id, call.name, {"path": "src/app.py"})


class MeteredModel:
    """每轮都要求继续检索，并声明固定的用量。"""

    def __init__(self, *, input_tokens: int, output_tokens: int, final_after: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.final_after = final_after
        self.rounds = 0

    def complete_with_tools(self, messages, tools):
        self.rounds += 1
        if self.rounds > self.final_after:
            return ToolModelResponse(
                "完成", (), "fake", self.input_tokens, self.output_tokens
            )
        return ToolModelResponse(
            None,
            # 每轮换一个路径：重复同一个调用会被「重复调用」闸先拦下，
            # 那样测的就不是花费闸了。
            (
                ToolCall(
                    f"c{self.rounds}", "read_file", {"path": f"src/app{self.rounds}.py"}
                ),
            ),
            "fake",
            self.input_tokens,
            self.output_tokens,
        )


def _budget_events(log: InMemoryEventLog, run_id: str, event_type: str):
    return [
        event for event in log.read_events(run_id) if event.event_type == event_type
    ]


def test_over_budget_stops_without_raising_and_records_the_fact():
    log = InMemoryEventLog()
    config = ToolLoopConfig(max_steps=8, token_budget=10_000)
    loop = ToolLoop(
        MeteredModel(input_tokens=5_000, output_tokens=100, final_after=99),
        Host(),
        config=config,
        run_id="run-budget",
        event_log=log,
    )

    result = loop.run("system", "question")

    assert result.status == "STUCK"
    assert result.reason == "Token 预算已用尽"
    assert result.budget_exhausted is True
    assert result.token_budget == 10_000
    assert result.tokens_used == 10_200  # 第二轮累计后越线，停在这里
    assert len(_budget_events(log, "run-budget", BUDGET_EXHAUSTED)) == 1
    assert len(_budget_events(log, "run-budget", BUDGET_LIMIT)) == 1
    used = _budget_events(log, "run-budget", BUDGET_USED)
    assert len(used) == 2
    assert used[-1].payload["budget_used"] == "10200"


def test_soft_threshold_fires_once_and_does_not_stop_the_run():
    log = InMemoryEventLog()
    config = ToolLoopConfig(max_steps=8, token_budget=100_000)
    loop = ToolLoop(
        MeteredModel(input_tokens=20_000, output_tokens=0, final_after=4),
        Host(),
        config=config,
        run_id="run-soft",
        event_log=log,
    )

    result = loop.run("system", "question")

    assert result.status == "COMPLETED"
    assert result.budget_exhausted is False
    flags = [
        event.payload["soft_threshold_reached"]
        for event in _budget_events(log, "run-soft", BUDGET_USED)
    ]
    # 70000 是 70% 线：每轮 20000，第 4 轮 80000 越线，第 5 轮 100000 收尾。
    # 100000 只是「用完」不是「超支」，所以这一轮不该被截断。
    assert flags == ["false", "false", "false", "true", "true"]


def test_without_a_declared_budget_nothing_is_recorded():
    log = InMemoryEventLog()
    loop = ToolLoop(
        MeteredModel(input_tokens=1_000_000, output_tokens=0, final_after=1),
        Host(),
        run_id="run-nobudget",
        event_log=log,
    )

    result = loop.run("system", "question")

    assert result.status == "COMPLETED"
    assert result.token_budget is None
    assert result.budget_exhausted is False
    assert _budget_events(log, "run-nobudget", BUDGET_LIMIT) == []
    assert _budget_events(log, "run-nobudget", BUDGET_USED) == []


def test_invalid_budget_configuration_is_rejected():
    with pytest.raises(ValueError):
        ToolLoopConfig(token_budget=0)
    with pytest.raises(ValueError):
        ToolLoopConfig(budget_soft_ratio=0.0)
    with pytest.raises(ValueError):
        ToolLoopConfig(budget_soft_ratio=1.0)
