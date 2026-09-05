from types import SimpleNamespace

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


def test_stdio_client_does_not_parse_text_as_tool_call():
    client = StdioMCPClient(".")
    assert client.server_module == "codeinsight.infrastructure.mcp_server"
