from pathlib import Path

from codeinsight.agent.change_workflow import run_change_workflow_stdio
from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse


class FixedSequenceModel:
    def __init__(self):
        self.round = 0

    def complete_with_tools(self, messages, tools):
        self.round += 1
        if self.round == 1:
            return ToolModelResponse(None, (ToolCall("c1", "get_repository_map", {}),), "fake")
        if self.round == 2:
            return ToolModelResponse(
                None, (ToolCall("c2", "search_repository", {"question": "checkout"}),), "fake"
            )
        return ToolModelResponse("已通过 MCP 获取地图和检索证据。", (), "fake")


def test_mainline_workflow_uses_real_stdio_mcp_client():
    root = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
    result = run_change_workflow_stdio(
        "理解 checkout 的实现位置", str(root), model=FixedSequenceModel()
    )
    assert result.mcp_transport == "stdio"
    assert result.loop.status == "COMPLETED"
    assert [item.tool_name for item in result.loop.tool_results] == [
        "get_repository_map",
        "search_repository",
    ]
