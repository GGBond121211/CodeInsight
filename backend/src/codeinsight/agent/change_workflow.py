"""Step 5 的代码任务入口：自主探索，但把修改交给受控执行层。"""

from __future__ import annotations

from dataclasses import dataclass

from codeinsight.agent.roles import planner_prompt
from codeinsight.agent.tool_loop import (
    MCPToolClient,
    ToolLoop,
    ToolLoopConfig,
    ToolLoopResult,
    ToolModel,
)
from codeinsight.application.change_service import ChangeService, PreviewResult
from codeinsight.infrastructure.mcp_client import StdioMCPClient


@dataclass(frozen=True)
class ChangeWorkflowResult:
    loop: ToolLoopResult
    mcp_transport: str = "stdio"


@dataclass(frozen=True)
class ChangeWorkflowPreviewResult:
    loop: ToolLoopResult
    preview: PreviewResult


def run_change_workflow_to_preview(
    task: str,
    repository_root: str,
    *,
    model: ToolModel,
    mcp_client: MCPToolClient,
    change_service: ChangeService,
    run_id: str,
    validation_profile: str = "python_compile",
    config: ToolLoopConfig | None = None,
) -> ChangeWorkflowPreviewResult:
    """把模型原生 ``generate_patch`` 调用接到受控 preview，不自动审批或应用。"""
    completed = run_change_workflow(
        task,
        model=model,
        mcp_client=mcp_client,
        config=config,
        run_id=run_id,
    )
    generated = [call for call in completed.loop.tool_calls if call.name == "generate_patch"]
    if completed.loop.status != "COMPLETED" or not generated:
        raise ValueError("Tool Loop 未生成可交付 Patch")
    changes: dict[str, str | None] = {}
    for call in generated:
        path = call.arguments.get("path")
        if not isinstance(path, str):
            raise ValueError("generate_patch 缺少合法 path")
        new_content = call.arguments.get("new_content")
        if not isinstance(new_content, str):
            raise ValueError("generate_patch 缺少合法 new_content")
        changes[path] = new_content
    preview = change_service.preview_many(
        repository_root,
        run_id=run_id,
        changes=changes,
        validation_profile=validation_profile,
    )
    return ChangeWorkflowPreviewResult(completed.loop, preview)


def run_change_workflow(
    task: str,
    *,
    model: ToolModel,
    mcp_client: MCPToolClient,
    config: ToolLoopConfig | None = None,
    run_id: str = "local-run",
    event_log=None,
) -> ChangeWorkflowResult:
    """运行一条真实 MCP 主链路。

    调用方显式传入 Client，既可以在集成环境传入 StdioMCPClient，也可以在
    单元测试中传 Fake Host；ToolLoop 仍然不会绕过 Client 调用内部 Python 函数。
    """
    system, user = planner_prompt(task)
    return ChangeWorkflowResult(
        ToolLoop(
            model,
            mcp_client,
            config=config,
            run_id=run_id,
            event_log=event_log,
        ).run(system, user)
    )


def run_change_workflow_stdio(
    task: str,
    repository_root: str,
    *,
    model: ToolModel,
    config: ToolLoopConfig | None = None,
    run_id: str = "local-run",
    event_log=None,
) -> ChangeWorkflowResult:
    with StdioMCPClient(repository_root) as client:
        return run_change_workflow(
            task,
            model=model,
            mcp_client=client,
            config=config,
            run_id=run_id,
            event_log=event_log,
        )
