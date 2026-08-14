"""CLI coverage for the Task 11 Smart Answer entry point."""

import json
from pathlib import Path

from codeinsight.cli.main import main
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch

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
    monkeypatch.setattr(
        "codeinsight.cli.main.OpenAIChatModel.from_environment",
        lambda: fake_model,
    )
    fake_embedding = type(
        "FakeEmbedding",
        (),
        {
            "embed": staticmethod(
                lambda texts: EmbeddingBatch("fake", tuple((1.0, 0.0) for _ in texts), len(texts))
            )
        },
    )()
    monkeypatch.setattr(
        "codeinsight.cli.main.OpenAIEmbeddingModel.from_environment",
        lambda: fake_embedding,
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
    assert "route: linear" in output
    assert "router: model=fake-router-and-answer tokens=12/7" in output
    assert "fallback: none" in output
