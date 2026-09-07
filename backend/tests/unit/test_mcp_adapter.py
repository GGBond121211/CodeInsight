from io import StringIO
from types import SimpleNamespace

import pytest

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.infrastructure.mcp_client import StdioMCPClient
from codeinsight.infrastructure.openai_chat import OpenAIChatModel


def test_openai_adapter_reads_native_tool_calls_only():
    function = SimpleNamespace(name="read_file", arguments='{"path":"src/a.py"}')
    message = SimpleNamespace(
        content=None, tool_calls=[SimpleNamespace(id="call-1", function=function)]
    )
    completions = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAIChatModel(client=client, model="fake")  # type: ignore[arg-type]
    result = model.complete_with_tools(
        [{"role": "user", "content": "read"}], [{"type": "function"}]
    )
    assert result.tool_calls == (ToolCall("call-1", "read_file", {"path": "src/a.py"}),)
    assert result.input_tokens == 7


@pytest.mark.parametrize("payload", ['[]', '{"error":[]}', '{"data":[]}'])
def test_stdio_client_rejects_malformed_tool_result(monkeypatch, payload):
    client = StdioMCPClient(".")
    monkeypatch.setattr(
        client, "_request",
        lambda *args: {"result": {"content": [{"text": payload}]}},
    )
    result = client.call_tool(ToolCall("x", "read_file", {"path": "x.py"}))
    assert not result.ok and result.error_code == "UNKNOWN"


def test_stdio_client_rejects_response_for_another_request():
    client = StdioMCPClient(".")
    client._stdin = StringIO()
    client._stdout = StringIO()
    client._responses.put('{"jsonrpc":"2.0","id":999,"result":{}}\n')

    with pytest.raises(RuntimeError, match="ID"):
        client._request("tools/list", {})


def test_stdio_client_bounds_wait_and_invalidates_connection(monkeypatch):
    client = StdioMCPClient(".", timeout_seconds=0.001)
    client._stdin = StringIO()
    client._stdout = StringIO()
    closed = []
    monkeypatch.setattr(client, "close", lambda: closed.append(True))
    with pytest.raises(RuntimeError):
        client._request("tools/list", {})
    assert closed == [True]
