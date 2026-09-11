"""Smart Answer CLI 入口测试。

2026-09-11 起 CLI 的 explain 也走只读 Tool Loop：Router 判 linear 或 agent
都收敛到同一条路，因此这里注入假的 MCP Client，不再注入 Embedding / Rerank。
"""

import json
from pathlib import Path

from codeinsight.cli.main import main
from codeinsight.domain.answer import ModelCompletion

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


class _FakeChatModel:
    model = "fake-router-and-answer"

    def __init__(self) -> None:
        self.tool_rounds = 0

    def complete(self, _system_prompt: str, _user_prompt: str) -> ModelCompletion:
        return ModelCompletion(
            json.dumps(
                {
                    "language": "en",
                    "normalized_question": "Where is checkout validation?",
                    "subquestions": [
                        {
                            "question": "Where is checkout validation?",
                            "intent": "implementation",
                            "retrieval_mode": "hybrid",
                        }
                    ],
                    "execution_route": "linear",
                    "confidence": 0.95,
                }
            ),
            self.model,
            12,
            7,
        )

    def generate(self, _system_prompt: str, _user_prompt: str):
        raise AssertionError("explain 路线不应调用 generate，只读 Tool Loop 走工具契约")

    def complete_with_tools(self, _messages, _tools):
        from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse

        self.tool_rounds += 1
        if self.tool_rounds == 1:
            return ToolModelResponse(
                None,
                (
                    ToolCall(
                        "call-1",
                        "search_repository",
                        {"question": "Where is checkout validation?"},
                    ),
                ),
                self.model,
                20,
                5,
            )
        return ToolModelResponse(
            '{"outcome":"answered","answer":'
            '"Checkout validation is implemented in the repository.",'
            '"citations":["E1","E2"]}',
            (),
            self.model,
            30,
            10,
        )


class _FakeMCPClient:
    """按项目契约返回两条命中，让证据评估判定为 sufficient。"""

    def __init__(self, root: str) -> None:
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def list_tools(self):
        from codeinsight.infrastructure.tool_registry import build_default_registry

        return build_default_registry().list_tools()

    def call_tool(self, call):
        from codeinsight.agent.tool_loop import ToolResult

        if call.name == "search_repository":
            return ToolResult.success(
                call.id,
                call.name,
                {
                    "results": [
                        {
                            "relative_path": "src/shop/validation.py",
                            "start_line": 1,
                            "end_line": 20,
                            "text": "def validate(): pass",
                            "rank": 1,
                        },
                        {
                            "relative_path": "src/shop/api.py",
                            "start_line": 5,
                            "end_line": 25,
                            "text": "def checkout(): pass",
                            "rank": 2,
                        },
                    ],
                    "untrusted": True,
                },
            )
        return ToolResult.success(call.id, call.name, {"ok": True})


def test_auto_answer_cli_runs_the_readonly_tool_loop(monkeypatch, capsys) -> None:
    fake_model = _FakeChatModel()

    def fake_chat_factory(*, context_assembler=None):
        return fake_model

    monkeypatch.setattr(
        "codeinsight.cli.main.OpenAIChatModel.from_environment",
        fake_chat_factory,
    )
    monkeypatch.setattr(
        "codeinsight.application.code_understanding_route.default_mcp_client_factory",
        lambda: _FakeMCPClient,
    )

    exit_code = main(
        [
            "auto-answer",
            "--repo",
            str(FIXTURE_ROOT),
            "Where is checkout validation?",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Checkout validation is implemented" in output
    assert "执行路线：linear" in output
    assert "Router：模型=fake-router-and-answer Token=12/7" in output
    assert "回退原因：无" in output
    assert "[E1] src/shop/validation.py:1-20" in output
    assert fake_model.tool_rounds == 2

