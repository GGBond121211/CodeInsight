import pytest

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk

# 2026-09-10：explain 已统一走只读 Tool Loop，本文件覆盖的是冻结的 LangGraph
# 实现。默认不运行（见 backend/pyproject.toml 的 addopts），复核历史实现时用
# pytest -m legacy。
pytestmark = pytest.mark.legacy


class CompletionSequence:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
        content = self.contents[self.calls]
        self.calls += 1
        return ModelCompletion(content, "fake-model", 10, 4)


def _search(*args, **kwargs) -> tuple[RankedChunk, ...]:
    return (
        RankedChunk(SourceChunk("src/api.py", 1, 5, "checkout(request)"), 2.0, 1),
        RankedChunk(SourceChunk("src/service.py", 1, 8, "def checkout(): pass"), 1.5, 2),
    )


def _empty_search(*args, **kwargs) -> tuple[RankedChunk, ...]:
    return ()


def test_no_evidence_stops_without_model_call() -> None:
    complete = CompletionSequence()

    result = run_citation_agent(
        "repo",
        "unknown",
        complete=complete,
        search=_empty_search,
        retrieval_mode="hybrid",
    )

    assert result.result.outcome == "insufficient_evidence"
    assert complete.calls == 0
    event_steps = []
    for event in result.events:
        event_steps.append(event.step)
    assert event_steps == ["retrieve", "finalize"]


def test_pass_review_prunes_unnecessary_citation_without_revision() -> None:
    complete = CompletionSequence(
        '{"outcome":"answered","answer":"Defined in service.","citations":["E1","E2"]}',
        '{"verdict":"pass","supported_citations":["E2"],"feedback":"E2 is direct."}',
    )

    result = run_citation_agent(
        "repo", "Where?", complete=complete, search=_search, retrieval_mode="hybrid"
    )

    citation_ids = []
    for citation in result.result.citations:
        citation_ids.append(citation.evidence_id)
    assert citation_ids == ["E2"]
    assert result.revisions == 0
    assert complete.calls == 2
    assert result.input_tokens == 20


def test_revise_route_runs_exactly_once_and_finishes() -> None:
    complete = CompletionSequence(
        '{"outcome":"answered","answer":"Defined in API and service.","citations":["E1","E2"]}',
        '{"verdict":"revise","supported_citations":["E2"],"feedback":"Remove API."}',
        '{"outcome":"answered","answer":"Defined in service.","citations":["E2"]}',
        '{"verdict":"pass","supported_citations":["E2"],"feedback":"Direct support."}',
    )

    result = run_citation_agent(
        "repo", "Where?", complete=complete, search=_search, retrieval_mode="hybrid"
    )

    assert result.revisions == 1
    assert complete.calls == 4
    event_steps = []
    for event in result.events:
        event_steps.append(event.step)
    assert event_steps == [
        "retrieve",
        "draft",
        "review",
        "revise",
        "review",
        "finalize",
    ]
    assert result.result.citations[0].relative_path == "src/service.py"


def test_insufficient_draft_skips_citation_review() -> None:
    complete = CompletionSequence(
        '{"outcome":"insufficient_evidence","answer":"Not supported.","citations":[]}'
    )

    result = run_citation_agent(
        "repo",
        "Which provider?",
        complete=complete,
        search=_search,
        retrieval_mode="hybrid",
    )

    assert result.result.outcome == "insufficient_evidence"
    assert result.result.citations == ()
    assert complete.calls == 1
    event_steps = []
    for event in result.events:
        event_steps.append(event.step)
    assert event_steps == ["retrieve", "draft", "finalize"]


def test_reviser_stops_after_five_bounded_rounds() -> None:
    draft = '{"outcome":"answered","answer":"Draft.","citations":["E1"]}'
    revise = '{"outcome":"answered","answer":"Revised.","citations":["E1"]}'
    review = '{"verdict":"revise","supported_citations":["E1"],"feedback":"Improve."}'
    complete = CompletionSequence(
        draft,
        review,
        revise,
        review,
        revise,
        review,
        revise,
        review,
        revise,
        review,
        revise,
        review,
    )

    result = run_citation_agent(
        "repo", "Where?", complete=complete, search=_search, retrieval_mode="hybrid"
    )

    assert result.revisions == 5
    assert complete.calls == 12
    revise_count = 0
    for event in result.events:
        if event.step == "revise":
            revise_count += 1
    assert revise_count == 5
