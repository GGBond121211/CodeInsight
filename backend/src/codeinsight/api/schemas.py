"""HTTP request and response schemas for the local CodeInsight API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["lexical", "bm25", "hybrid"]
RetrievalReason = Literal[
    "direct_match",
    "semantic_match",
    "hybrid_match",
]


class SearchRequest(BaseModel):
    repository_root: str = Field(min_length=1)
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = "hybrid"

    @field_validator("repository_root", "question")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class AnswerRequest(BaseModel):
    repository_root: str = Field(min_length=1)
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = "hybrid"

    @field_validator("repository_root", "question")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class AgentAnswerRequest(AnswerRequest):
    retrieval_mode: RetrievalMode = "hybrid"


class AutoAnswerRequest(BaseModel):
    repository_root: str = Field(min_length=1)
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    force_route: Literal["linear", "agent"] | None = None

    @field_validator("repository_root", "question")
    @classmethod
    def strip_auto_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class SearchHitResponse(BaseModel):
    rank: int
    score: float
    relative_path: str
    start_line: int
    end_line: int
    excerpt: str
    symbol_path: str | None
    retrieval_reason: RetrievalReason


class SearchResponse(BaseModel):
    retrieval_mode: RetrievalMode
    results: list[SearchHitResponse]


class CitationResponse(BaseModel):
    evidence_id: str
    relative_path: str
    start_line: int
    end_line: int


class TokenUsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int


class AnswerResponse(BaseModel):
    outcome: str
    answer: str
    citations: list[CitationResponse]
    retrieval_mode: RetrievalMode
    model: str | None
    prompt_version: str
    usage: TokenUsageResponse


class AgentEventResponse(BaseModel):
    sequence: int
    step: str
    summary: str


class AgentAnswerResponse(AnswerResponse):
    revisions: int
    events: list[AgentEventResponse]


class QueryPlanSubQuestionResponse(BaseModel):
    question: str
    intent: str
    retrieval_mode: RetrievalMode


class QueryPlanResponse(BaseModel):
    original_question: str
    language: str
    normalized_question: str
    subquestions: list[QueryPlanSubQuestionResponse]
    retrieval_modes: list[RetrievalMode]
    execution_route: Literal["linear", "agent", "insufficient"]
    confidence: float
    fallback_reason: str | None


class AutoSubQuestionResponse(BaseModel):
    question: str
    intent: str
    retrieval_mode: RetrievalMode
    outcome: str
    answer: str
    citations: list[CitationResponse]


class AutoEventResponse(BaseModel):
    sequence: int
    step: str
    summary: str


class AutoAnswerResponse(BaseModel):
    outcome: str
    answer: str
    citations: list[CitationResponse]
    retrieval_mode: str
    model: str | None
    prompt_version: str
    usage: TokenUsageResponse
    plan: QueryPlanResponse
    subquestions: list[AutoSubQuestionResponse]
    router_model: str | None
    router_usage: TokenUsageResponse
    router_elapsed_milliseconds: float
    embedding_input_tokens: int
    fallback_reason: str | None
    events: list[AutoEventResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
