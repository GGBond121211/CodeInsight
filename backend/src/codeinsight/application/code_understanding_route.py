"""explain 路线的唯一入口：只读代码理解 Tool Loop（Q-008）。

2026-09-10 起，旧的 LangGraph 路线（``agent/workflow.py:run_citation_agent``）
已停用：代码保留在仓库里作为历史实现，但没有任何运行入口再调用它，
``explain` 一律走这里的只读 Tool Loop。

2026-09-11 起，Router 的 ``linear`` 分支也不再走「直接回答」：
``linear`` 与 ``agent`` 都收敛到只读 Tool Loop，只有 ``insufficient``
不检索。用户的决定是产品决定，不是实验结果——本文不声称新路质量
优于被替换的路线。

为什么把三个入口收成一个函数：
    对话（conversation_service）、HTTP（api/routes）、CLI（cli/main）原本各自
    抄了一份 ``execution_route == "agent"` 分支。三份分支意味着三次漂移机会——
    关闭老路时只要漏掉其中一处，就会留下一条仍然走 LangGraph 的暗门。
    现在三处共用同一实现，不存在「某处还没切过来」的状态。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from codeinsight.agent.code_understanding_tool_loop import (
    ANSWERED_STATUS,
    CodeUnderstandingResult,
    CodeUnderstandingToolLoop,
)
from codeinsight.application.query_router import QueryRouterResult
from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    AutoAnswer,
    AutoAnswerEvent,
    SubQuestionAnswer,
)

MCPClientFactory = Callable[[str], object]

# explain 的唯一路线集合：linear 与 agent 都走只读 Tool Loop，只有
# insufficient 不检索。判断集中在这里，三条入口共用同一份事实。
CODE_UNDERSTANDING_ROUTES: frozenset[str] = frozenset({"linear", "agent"})


def uses_code_understanding(execution_route: str) -> bool:
    """explain 是否要跑只读 Tool Loop（``insufficient`` 除外）。"""
    return execution_route in CODE_UNDERSTANDING_ROUTES


def default_mcp_client_factory() -> MCPClientFactory:
    """生产环境的 MCP Client：每个 Run 起一个 stdio 子进程。"""
    from codeinsight.infrastructure.mcp_client import StdioMCPClient

    return StdioMCPClient


def run_code_understanding_answer(
    repository_root: str | Path,
    question: str,
    *,
    model: object,
    run_id: str = "local-run",
    event_log=None,
    mcp_client_factory: MCPClientFactory | None = None,
    emit=None,
) -> CodeUnderstandingResult:
    """跑一次只读代码理解 Tool Loop。

    ``model`` 必须具备 ``complete_with_tools``；只读白名单、证据台账和
    Repair 预算都在 Tool Loop 内部生效，调用方不需要也无法放宽。
    """
    factory = mcp_client_factory or default_mcp_client_factory()
    root = str(Path(repository_root).resolve())
    with factory(root) as client:
        loop = CodeUnderstandingToolLoop(
            model,
            client,
            run_id=run_id,
            event_log=event_log,
            repo_root=root,
            emit=emit,
        )
        return loop.run(question)


def to_auto_answer(
    result: CodeUnderstandingResult,
    router_result: QueryRouterResult,
    *,
    model_name: str | None,
    retrieval_mode: str = "auto",
) -> AutoAnswer:
    """把 Tool Loop 结果映射成公开的 AutoAnswer 形状。

    只有 ``ANSWERED`` 映射成 answered；其余状态（证据不足、部分回答、
    超时、卡住、失败）一律映射成 insufficient_evidence，避免把未验证的
    结论以「已回答」的身份交给用户。
    """
    outcome = ANSWERED if result.status == ANSWERED_STATUS else INSUFFICIENT_EVIDENCE
    subquestions = tuple(
        SubQuestionAnswer(
            question=item.question,
            intent=item.intent,
            # 每个子问题沿用 Router 规划时选定的检索模式；Tool Loop 内部
            # 可能换用别的模式补证据，所以这里只声明「计划模式」，不假装
            # 知道每一次工具调用实际用了什么。
            retrieval_mode=item.retrieval_mode,
            outcome=outcome,
            answer=result.answer,
            citations=result.citations,
        )
        for item in router_result.plan.subquestions
    )
    return AutoAnswer(
        outcome=outcome,
        answer=result.answer,
        citations=result.citations,
        retrieval_mode=retrieval_mode,
        model=model_name,
        prompt_version=result.prompt_version,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        # MCP Server 在子进程内做 Embedding，用量目前不回流主进程。
        embedding_input_tokens=0,
        plan=router_result.plan,
        subquestions=subquestions,
        router_model=router_result.model,
        router_input_tokens=router_result.input_tokens,
        router_output_tokens=router_result.output_tokens,
        router_elapsed_milliseconds=router_result.elapsed_milliseconds,
        fallback_reason=router_result.fallback_reason,
        events=_public_events(result),
    )


def _public_events(result: CodeUnderstandingResult) -> tuple[AutoAnswerEvent, ...]:
    """只暴露确定性计数和状态，不含隐藏推理或工具输出原文。"""
    events = [
        AutoAnswerEvent(1, "tool_loop", f"只读工具调用 {result.tool_calls} 次，{result.steps} 步"),
        AutoAnswerEvent(
            2,
            "evidence",
            f"可引用证据 {len(result.evidence)} 条，评估为 {result.assessment.status}",
        ),
    ]
    if result.repair_rounds:
        events.append(
            AutoAnswerEvent(
                3,
                "repair",
                f"证据修复 {result.repair_rounds} 轮：{','.join(result.repair_actions)}",
            )
        )
    events.append(
        AutoAnswerEvent(
            len(events) + 1,
            "finalize",
            f"终止状态 {result.status}"
            + (f"（{result.termination_reason}）" if result.termination_reason else ""),
        )
    )
    return tuple(events)
