from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


class FakeModel:
    model = "fake-router"

    def __init__(self) -> None:
        # 2026-09-11 起 explain 都走只读 Tool Loop：先检索，再给最终答案。
        self.tool_rounds = 0

    def complete(self, _system_prompt: str, _user_prompt: str) -> ModelCompletion:
        return ModelCompletion(
            '{"language":"en","normalized_question":"Where is checkout?",'
            '"subquestions":[{"question":"Where is checkout?","intent":"symbol_lookup",'
            '"retrieval_mode":"hybrid"}],"execution_route":"linear","confidence":0.9}',
            "fake-router",
            5,
            2,
        )

    def generate(self, _system_prompt: str, _user_prompt: str) -> ModelAnswer:
        return ModelAnswer("answered", "checkout is defined here.", ("E1",), "fake-answer", 8, 3)

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
                        {"question": "Where is checkout?"},
                    ),
                ),
                self.model,
            )
        return ToolModelResponse(
            '{"outcome":"answered","answer":"checkout is defined here.",'
            '"citations":["E1","E2"]}',
            (),
            self.model,
        )


class FakeEmbedding:
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


class FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


class FakeMCPClient:
    """只读 Tool Loop 的假 MCP Client：两条命中，证据评估判 sufficient。"""

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


def test_health_auto_answer_and_usage_are_the_registered_product_routes() -> None:
    client = TestClient(
        create_app(
            FakeModel,
            FakeEmbedding,
            reranker_factory=FakeReranker,
            mcp_client_factory=FakeMCPClient,
        )
    )  # type: ignore[arg-type]

    assert client.get("/api/v1/health").status_code == 200
    auto = client.post(
        "/api/v1/auto/answer",
        json={"repository_root": str(FIXTURE_ROOT), "question": "Where is checkout?"},
    )
    assert auto.status_code == 200
    assert auto.json()["outcome"] == "answered"

    for path in ("/api/v1/search", "/api/v1/answer", "/api/v1/agent/answer"):
        assert client.post(path, json={}).status_code == 404


def test_openapi_exposes_current_product_routes() -> None:
    client = TestClient(
        create_app(FakeModel, FakeEmbedding, reranker_factory=FakeReranker)
    )  # type: ignore[arg-type]
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/api/v1/health",
        "/api/v1/auto/answer",
        "/api/v1/usage/summary",
        "/api/v1/usage/calls",
        "/api/v2/change/preview",
        "/api/v2/change/approve",
        "/api/v2/change/apply",
        "/api/v2/change/rollback",
        "/api/v2/change/{run_id}/cancel",
        "/api/v2/change/{run_id}/events",
        "/api/v2/change/{run_id}/patches/{patch_id}",
        "/api/v2/chat/sessions",
        "/api/v2/chat/sessions/{session_id}",
        "/api/v2/chat/turns",
        "/api/v2/chat/turns/{turn_id}",
        "/api/v2/chat/runs/{run_id}",
        "/api/v2/chat/turns/{turn_id}/approve",
        "/api/v2/chat/turns/{turn_id}/cancel",
        "/api/v2/chat/turns/{turn_id}/events",
    }
