"""HTTP contract tests for the optional Smart Answer endpoint.

2026-09-11 起 explain 只有一条路线：Router 判 linear 或 agent 都收敛到只读
MCP Tool Loop（`application/code_understanding_route.py`），只有 insufficient
不检索、不调用工具。下面几个用例分别守住这三件事。
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_ROOT = BACKEND_ROOT / "backend" / "tests" / "fixtures" / "sample_repo"


class _FakeEmbedding:
    @staticmethod
    def embed(texts):
        vectors = []
        for _ in texts:
            vectors.append((1.0, 0.0))
        return EmbeddingBatch(
            "fake",
            tuple(vectors),
            len(texts),
            tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
            "dense-sparse-v1",
        )


class _FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


# ---------------------------------------------------------------------------
# 旧 linear「直接回答」路线的假模型（2026-09-11 起 explain 不再走它）
#
# 保留以便回退；当前用例不再使用这几个定义。
# ---------------------------------------------------------------------------
# class _FakeAutoModel:
#     def __init__(self, router_content: str) -> None:
#         self.router_content = router_content
#         self.router_calls = 0
#         self.answer_calls = 0
#
#     def complete(self, _system: str, _user: str) -> ModelCompletion:
#         self.router_calls += 1
#         return ModelCompletion(self.router_content, "fake-router", 7, 3)
#
#     def generate(self, _system: str, _user: str) -> ModelAnswer:
#         self.answer_calls += 1
#         return ModelAnswer(
#             "answered",
#             "checkout validates input.",
#             ("E1",),
#             "fake-answer",
#             12,
#             5,
#         )


def _router_payload(execution_route: str, **overrides: object) -> str:
    payload: dict[str, object] = {
        "language": "en",
        "normalized_question": "Explain validation and pricing.",
        "subquestions": [
            {
                "question": "How is input validated?",
                "intent": "implementation",
                "retrieval_mode": "hybrid",
            }
        ],
        "execution_route": execution_route,
        "confidence": 0.95,
    }
    payload.update(overrides)
    return json.dumps(payload)


class _ToolLoopModel:
    """脚本模型：第一轮检索，第二轮给最终答案；Router 也由它回答。"""

    def __init__(self, router_content: str) -> None:
        self.router_content = router_content
        self.model = "fake-tool-loop"
        self.tool_rounds = 0

    def complete(self, _system: str, _user: str) -> ModelCompletion:
        return ModelCompletion(self.router_content, "fake-router", 7, 3)

    def generate(self, _system: str, _user: str) -> ModelAnswer:
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
                        {"question": "How is input validated?"},
                    ),
                ),
                self.model,
                20,
                5,
            )
        return ToolModelResponse(
            '{"outcome":"answered","answer":"Validation happens in checkout.",'
            '"citations":["E1","E2"]}',
            (),
            self.model,
            30,
            10,
        )


class _ToolLoopClient:
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
                            "relative_path": "src/shop/checkout.py",
                            "start_line": 1,
                            "end_line": 30,
                            "text": "def validate(): pass",
                            "rank": 1,
                        },
                        {
                            "relative_path": "src/shop/pricing.py",
                            "start_line": 1,
                            "end_line": 20,
                            "text": "def price(): pass",
                            "rank": 2,
                        },
                    ],
                    "untrusted": True,
                },
            )
        return ToolResult.success(call.id, call.name, {"ok": True})


def _client(model: _ToolLoopModel) -> TestClient:
    return TestClient(
        create_app(
            lambda: model,
            _FakeEmbedding,
            reranker_factory=_FakeReranker,
            mcp_client_factory=_ToolLoopClient,
        )
    )  # type: ignore[arg-type]


def _ask(client: TestClient, question: str) -> dict:
    response = client.post(
        "/api/v1/auto/answer",
        json={"repository_root": str(FIXTURE_ROOT), "question": question},
    )
    assert response.status_code == 200
    return response.json()


def test_linear_plan_also_runs_the_readonly_tool_loop() -> None:
    """Router 判 linear 也收敛到只读 Tool Loop（2026-09-11 收口）。"""
    model = _ToolLoopModel(_router_payload("linear"))
    payload = _ask(_client(model), "How is input validated?")

    assert payload["plan"]["execution_route"] == "linear"
    assert payload["outcome"] == "answered"
    assert payload["answer"] == "Validation happens in checkout."
    assert payload["citations"][0]["relative_path"] == "src/shop/checkout.py"
    # 证据编号由应用分配，模型只返回编号。
    assert payload["citations"][0]["evidence_id"] == "E1"
    assert payload["usage"] == {"input_tokens": 50, "output_tokens": 15}
    # Tool Loop 的 Embedding 在 MCP Server 子进程内发生，父进程不回流用量。
    assert payload["embedding_input_tokens"] == 0
    assert payload["router_usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert model.tool_rounds == 2
    assert payload["events"]


def test_agent_plan_runs_the_readonly_tool_loop() -> None:
    """agent 路线同样走只读 Tool Loop，每个 Router 子问题都得到映射。"""
    model = _ToolLoopModel(
        _router_payload(
            "agent",
            subquestions=[
                {
                    "question": "How is input validated?",
                    "intent": "implementation",
                    "retrieval_mode": "hybrid",
                },
                {
                    "question": "How is price computed?",
                    "intent": "data_flow",
                    "retrieval_mode": "hybrid",
                },
            ],
        )
    )
    payload = _ask(_client(model), "Explain validation and pricing.")

    assert payload["plan"]["execution_route"] == "agent"
    assert payload["outcome"] == "answered"
    assert len(payload["subquestions"]) == 2
    assert {item["outcome"] for item in payload["subquestions"]} == {"answered"}
    assert model.tool_rounds == 2


def test_router_fallback_plan_also_runs_the_tool_loop() -> None:
    """Router 返回无效 JSON 时回退成 linear 计划，同样收敛到 Tool Loop。"""
    model = _ToolLoopModel("not-json")
    payload = _ask(_client(model), "Where is checkout defined?")

    assert payload["plan"]["execution_route"] == "linear"
    assert payload["plan"]["retrieval_modes"] == ["hybrid"]
    assert payload["fallback_reason"] == "router_invalid_or_low_confidence"
    assert payload["outcome"] == "answered"
    assert model.tool_rounds == 2


def test_insufficient_plan_skips_retrieval_and_tools() -> None:
    """insufficient 不检索、不调用工具，也不调用 generate。"""
    model = _ToolLoopModel(_router_payload("insufficient", subquestions=[]))
    payload = _ask(_client(model), "今天天气怎么样？")

    assert payload["plan"]["execution_route"] == "insufficient"
    assert payload["outcome"] == "insufficient_evidence"
    assert payload["subquestions"] == []
    assert payload["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert model.tool_rounds == 0

