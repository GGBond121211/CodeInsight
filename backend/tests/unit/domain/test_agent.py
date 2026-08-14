from dataclasses import FrozenInstanceError

import pytest

from codeinsight.domain.agent import AgentEvent, AgentRepositoryAnswer, CitationReview
from codeinsight.domain.answer import RepositoryAnswer


def test_agent_values_are_immutable() -> None:
    answer = RepositoryAnswer(
        outcome="insufficient_evidence",
        answer="Not enough evidence.",
        citations=(),
        retrieval_mode="bm25",
        model="test-model",
        prompt_version="citation-agent-v3",
        input_tokens=10,
        output_tokens=5,
    )
    result = AgentRepositoryAnswer(
        result=answer,
        revisions=0,
        events=(AgentEvent(1, "retrieve", "Found no evidence."),),
        input_tokens=10,
        output_tokens=5,
    )

    with pytest.raises(FrozenInstanceError):
        result.revisions = 1  # type: ignore[misc]


def test_review_holds_supported_evidence_and_public_feedback() -> None:
    review = CitationReview("revise", ("E2",), "Remove the unrelated call site.")

    assert review.supported_evidence_ids == ("E2",)
    assert review.feedback.startswith("Remove")
