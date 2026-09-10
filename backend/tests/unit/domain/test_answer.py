from dataclasses import FrozenInstanceError

import pytest

from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    AnswerCitation,
    ModelAnswer,
    RepositoryAnswer,
)


def test_answer_values_are_immutable() -> None:
    citation = AnswerCitation("E1", "src/app.py", 3, 8)
    result = RepositoryAnswer(
        outcome=ANSWERED,
        answer="The implementation is in app.py.",
        citations=(citation,),
        retrieval_mode="hybrid",
        model="test-model",
        prompt_version="code-answer-v1",
        input_tokens=10,
        output_tokens=5,
    )

    with pytest.raises(FrozenInstanceError):
        result.answer = "changed"  # type: ignore[misc]


def test_model_answer_holds_evidence_ids_and_usage() -> None:
    answer = ModelAnswer(
        outcome=INSUFFICIENT_EVIDENCE,
        answer="The evidence does not identify a provider.",
        evidence_ids=(),
        model="test-model",
        input_tokens=12,
        output_tokens=7,
    )

    assert answer.evidence_ids == ()
    assert answer.input_tokens == 12
