"""HTTP contract tests for the optional Smart Answer endpoint."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.answer import (
    ModelAnswer,
    ModelCompletion,
)
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


class _FakeAutoModel:
    def __init__(self, router_content: str) -> None:
        self.router_content = router_content
        self.router_calls = 0
        self.answer_calls = 0

    def complete(self, _system: str, _user: str) -> ModelCompletion:
        self.router_calls += 1
        return ModelCompletion(self.router_content, "fake-router", 7, 3)

    def generate(self, _system: str, _user: str) -> ModelAnswer:
        self.answer_calls += 1
        return ModelAnswer(
            "answered",
            "checkout validates input.",
            ("E1",),
            "fake-answer",
            12,
            5,
        )


def _router_payload() -> str:
    return json.dumps(
        {
            "language": "en",
            "normalized_question": "Where is checkout validation implemented?",
            "subquestions": [
                {
                    "question": "Where is checkout validation implemented?",
                    "intent": "implementation",
                    "retrieval_mode": "hybrid",
                }
            ],
            "execution_route": "linear",
            "confidence": 0.92,
        }
    )


def _model_factory(model):
    def factory():
        return model

    return factory


def test_auto_answer_returns_plan_subquestion_and_router_usage() -> None:
    model = _FakeAutoModel(_router_payload())
    client = TestClient(
        create_app(_model_factory(model), _FakeEmbedding, reranker_factory=_FakeReranker)
    )  # type: ignore[arg-type]

    response = client.post(
        "/api/v1/auto/answer",
        json={
            "repository_root": str(FIXTURE_ROOT),
            "question": "Where is checkout validation implemented?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["execution_route"] == "linear"
    assert payload["plan"]["subquestions"][0]["retrieval_mode"] == "hybrid"
    assert payload["subquestions"][0]["outcome"] == "answered"
    assert payload["router_usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert payload["usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert payload["embedding_input_tokens"] > 0
    assert model.router_calls == 1
    assert model.answer_calls == 1


def test_auto_answer_invalid_router_output_uses_linear_hybrid_fallback() -> None:
    model = _FakeAutoModel("not-json")
    client = TestClient(
        create_app(_model_factory(model), _FakeEmbedding, reranker_factory=_FakeReranker)
    )  # type: ignore[arg-type]

    response = client.post(
        "/api/v1/auto/answer",
        json={
            "repository_root": str(FIXTURE_ROOT),
            "question": "Where is checkout defined?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["execution_route"] == "linear"
    assert payload["plan"]["retrieval_modes"] == ["hybrid"]
    assert payload["fallback_reason"] == "router_invalid_or_low_confidence"
    assert model.answer_calls == 1


class _AgentRouteModel:
    """agent 路线用的脚本模型：第一轮检索，第二轮给最终答案。"""

    def __init__(self, router_content: str) -> None:
        self.router_content = router_content
        self.model = "fake-agent-route"
        self.tool_rounds = 0

    def complete(self, _system: str, _user: str) -> ModelCompletion:
        return ModelCompletion(self.router_content, "fake-router", 7, 3)

    def generate(self, _system: str, _user: str) -> ModelAnswer:
        raise AssertionError("agent 路线不应调用 generate")

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


class _AgentRouteClient:
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


def test_agent_route_runs_the_readonly_tool_loop() -> None:
    """2026-09-10 起 agent 路线走只读 Tool Loop，不再走旧 LangGraph。"""
    router_payload = json.dumps(
        {
            "language": "en",
            "normalized_question": "Explain validation and pricing.",
            "subquestions": [
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
            "execution_route": "agent",
            "confidence": 0.95,
        }
    )
    model = _AgentRouteModel(router_payload)
    client = TestClient(
        create_app(
            _model_factory(model),
            _FakeEmbedding,
            reranker_factory=_FakeReranker,
            mcp_client_factory=_AgentRouteClient,
        )
    )  # type: ignore[arg-type]

    response = client.post(
        "/api/v1/auto/answer",
        json={
            "repository_root": str(FIXTURE_ROOT),
            "question": "Explain validation and pricing.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "answered"
    assert payload["citations"][0]["relative_path"] == "src/shop/checkout.py"
    # 证据编号由应用分配，模型只返回编号。
    assert payload["citations"][0]["evidence_id"] == "E1"
    # 每个 Router 子问题都得到一条映射，但内容来自同一次只读运行。
    assert len(payload["subquestions"]) == 2
    assert {item["outcome"] for item in payload["subquestions"]} == {"answered"}
    assert payload["usage"]["input_tokens"] == 20 + 30
    assert model.tool_rounds == 2
