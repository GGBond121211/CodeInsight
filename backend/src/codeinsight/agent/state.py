"""Internal LangGraph state for the citation refinement Agent."""

from pathlib import Path
from typing import TypedDict

from codeinsight.domain.agent import AgentEvent, AgentRepositoryAnswer, CitationReview
from codeinsight.domain.answer import ModelAnswer
from codeinsight.domain.query_plan import QueryPlan
from codeinsight.domain.retrieval import RankedChunk, SubQuestionEvidence


class CitationAgentState(TypedDict, total=False):
    repository_root: str | Path
    question: str
    limit: int
    retrieval_mode: str
    query_plan: QueryPlan
    route_reason: str
    router_confidence: float
    subquestion_results: tuple[tuple[str, tuple[RankedChunk, ...]], ...]
    subquestion_evidence: tuple[SubQuestionEvidence, ...]
    results: tuple[RankedChunk, ...]
    draft: ModelAnswer
    subquestion_drafts: tuple[ModelAnswer, ...]
    review: CitationReview
    subquestion_reviews: tuple[CitationReview | None, ...]
    subquestion_accepted: tuple[bool, ...]
    revised: ModelAnswer
    revisions: int
    max_revisions: int
    events: tuple[AgentEvent, ...]
    input_tokens: int
    output_tokens: int
    agent_result: AgentRepositoryAnswer
