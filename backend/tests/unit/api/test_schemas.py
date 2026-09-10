import pytest
from pydantic import ValidationError

from codeinsight.api.schemas import (
    AgentAnswerRequest,
    AnswerRequest,
    SearchHitResponse,
    SearchRequest,
)


def test_request_defaults_use_hybrid_for_normal_product_paths() -> None:
    search = SearchRequest(repository_root=" repo ", question=" question ")
    answer = AnswerRequest(repository_root=" repo ", question=" question ")
    agent_answer = AgentAnswerRequest(repository_root=" repo ", question=" question ")

    assert search.repository_root == "repo"
    assert search.question == "question"
    assert search.retrieval_mode == "hybrid"
    assert answer.retrieval_mode == "hybrid"
    assert agent_answer.retrieval_mode == "hybrid"


@pytest.mark.parametrize("limit", [0, 21])
def test_request_limit_is_bounded(limit: int) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(repository_root="repo", question="question", limit=limit)


@pytest.mark.parametrize("field", ["repository_root", "question"])
def test_required_text_rejects_blank_values(field: str) -> None:
    values = {"repository_root": "repo", "question": "question", field: "   "}

    with pytest.raises(ValidationError):
        AnswerRequest(**values)


def test_hybrid_retrieval_and_source_context_are_part_of_internal_contract() -> None:
    request = SearchRequest(
        repository_root="repo",
        question="question",
        retrieval_mode="hybrid",
    )
    hit = SearchHitResponse(
        rank=1,
        score=2.0,
        relative_path="app.py",
        start_line=4,
        end_line=6,
        excerpt="def run():",
        symbol_path="Service.run",
        retrieval_reason="direct_match",
    )

    assert request.retrieval_mode == "hybrid"
    assert hit.symbol_path == "Service.run"
    assert hit.retrieval_reason == "direct_match"


@pytest.mark.parametrize("removed_mode", ["keyword", "unsupported"])
def test_removed_retrieval_modes_are_rejected(removed_mode: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(repository_root="repo", question="question", retrieval_mode=removed_mode)
