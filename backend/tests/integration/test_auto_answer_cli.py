"""Task 11 Smart Answer CLI 入口测试。"""

import json
from pathlib import Path

from codeinsight.cli.main import main
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


class _FakeChatModel:
    model = "fake-router-and-answer"

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
                            "retrieval_mode": "bm25",
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

    def generate(self, _system_prompt: str, _user_prompt: str) -> ModelAnswer:
        return ModelAnswer(
            "answered",
            "Checkout validation is implemented in the repository.",
            ("E1",),
            self.model,
            30,
            10,
        )


def test_auto_answer_cli_routes_and_prints_public_metadata(monkeypatch, capsys) -> None:
    fake_model = _FakeChatModel()

    def fake_chat_factory(*, context_assembler=None):
        return fake_model

    monkeypatch.setattr(
        "codeinsight.cli.main.OpenAIChatModel.from_environment",
        fake_chat_factory,
    )

    class FakeEmbedding:
        @staticmethod
        def embed(texts):
            vectors = []
            for _ in texts:
                vectors.append((1.0, 0.0))
            return EmbeddingBatch("fake", tuple(vectors), len(texts))

    fake_embedding = FakeEmbedding()

    def fake_embedding_factory():
        return fake_embedding

    monkeypatch.setattr(
        "codeinsight.cli.main.OpenAIEmbeddingModel.from_environment",
        fake_embedding_factory,
    )

    class FakeReranker:
        @staticmethod
        def rerank(_query, documents, *, top_n):
            return tuple(
                RerankResult(index=index, relevance_score=float(len(documents) - index))
                for index in range(top_n)
            )

    monkeypatch.setattr(
        "codeinsight.cli.main.OpenAITextReranker.from_environment",
        lambda: FakeReranker(),
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
