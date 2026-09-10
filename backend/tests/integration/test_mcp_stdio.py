from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.infrastructure.mcp_client import StdioMCPClient


def test_real_mcp_stdio_discovers_tools_and_executes_through_executor():
    root = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
    with StdioMCPClient(root) as client:
        tools = client.list_tools()
        names = {str(item["name"]) for item in tools}
        result = client.call_tool(ToolCall("read-1", "read_file", {"path": "src/shop/models.py"}))
    assert len(names) == 10
    assert "search_repository" in names
    assert {"lsp_definition", "scip_references"} <= names
    assert result.ok
    assert result.data["path"] == "src/shop/models.py"


def test_concurrent_stdio_calls_keep_their_responses(tmp_path):
    for name in ("first.py", "second.py"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    with StdioMCPClient(tmp_path) as client, ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            ToolCall(name, "read_file", {"path": name})
            for name in ("first.py", "second.py")
        ]
        results = list(pool.map(client.call_tool, calls))
    assert all(result.ok for result in results)
    assert [result.data["text"] for result in results] == ["first.py", "second.py"]


def test_real_mcp_stdio_rejects_escape_and_write_without_capabilities():
    with StdioMCPClient(Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo") as client:
        outside = client.call_tool(ToolCall("read-2", "read_file", {"path": "../secret.txt"}))
        write = client.call_tool(
            ToolCall(
                "write-1",
                "apply_patch_isolated",
                {"patch_id": "p", "approval_token": "a", "checkpoint_id": "c"},
            )
        )
    assert not outside.ok and outside.error_code == "PERMISSION"
    assert not write.ok
    assert write.error_code == "VALIDATION"


def test_real_mcp_resources_prompts_and_server_failure_are_explainable():
    client = StdioMCPClient(Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo")
    client.start()
    resources = client.list_resources()
    prompts = client.list_prompts()
    assert {item["uri"] for item in resources} == {
        "repo://manifest",
        "repo://evidence-context",
    }
    assert {item["name"] for item in prompts} == {"planner", "patch-reviewer"}
    assert "planner-v1" in client.get_prompt("planner")
    assert "untrusted" in client.read_resource("repo://manifest")
    client.close()
    failed = client.call_tool(ToolCall("closed", "read_file", {"path": "src/shop/models.py"}))
    assert not failed.ok and failed.error_code == "UNKNOWN"
