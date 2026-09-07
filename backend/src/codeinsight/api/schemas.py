"""本地 CodeInsight API 的 HTTP 请求和响应 Schema。"""

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
            raise ValueError("不能为空")
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
            raise ValueError("不能为空")
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
            raise ValueError("不能为空")
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


class ChangePreviewRequest(BaseModel):
    repository_root: str = Field(min_length=1)
    run_id: str | None = None
    path: str = Field(min_length=1)
    new_content: str
    validation_profile: str = "python_compile"

    @field_validator("repository_root", "path", "validation_profile")
    @classmethod
    def strip_change_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空")
        return stripped


class ChangePreviewResponse(BaseModel):
    status: Literal["preview_ready"] = "preview_ready"
    run_id: str
    patch_id: str
    path: str
    diff: str
    diff_hash: str
    base_fingerprint: str
    validation_profile: str
    requires_approval: bool = True


class ChangeApproveRequest(BaseModel):
    run_id: str = Field(min_length=1)
    patch_id: str = Field(min_length=1)
    expires_in_seconds: int = Field(default=300, ge=30, le=3600)


class ChangeApproveResponse(BaseModel):
    status: Literal["approval_granted"] = "approval_granted"
    run_id: str
    patch_id: str
    approval_token: str
    expires_in_seconds: int


class ChangeApplyRequest(BaseModel):
    run_id: str = Field(min_length=1)
    patch_id: str = Field(min_length=1)
    approval_token: str = Field(min_length=1)


class ChangeResultResponse(BaseModel):
    status: str
    run_id: str
    patch_id: str
    diff: str
    diff_hash: str
    base_fingerprint: str
    workspace_id: str | None = None
    checkpoint_id: str | None = None
    validation: dict[str, object] | None = None
    reason: str | None = None


class ChangeRollbackRequest(BaseModel):
    run_id: str = Field(min_length=1)
    patch_id: str = Field(min_length=1)


class ChangeCancelResponse(BaseModel):
    status: Literal["cancel_requested", "already_finished"]
    run_id: str
    immediate: bool


class ChangeEventResponse(BaseModel):
    event_id: str
    run_id: str
    sequence: int
    event_type: str
    occurred_at_epoch_ms: int
    payload: dict[str, str]


class ChatSessionRequest(BaseModel):
    repository_root: str = Field(min_length=1)
    session_id: str | None = None

    @field_validator("repository_root", "session_id")
    @classmethod
    def strip_session_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空")
        return stripped


class ChatSessionResponse(BaseModel):
    session_id: str
    repo_id: str
    index_version: str
    status: str = "READY"
    summary: str | None
    compacted_through_sequence: int
    active_goal: dict[str, object] | None
    recent_turns: list[dict[str, object]]
    cache_hit: bool
    cache_fallback: bool


class ChatTurnRequest(BaseModel):
    session_id: str | None = None
    repository_root: str = Field(min_length=1)
    message: str = Field(min_length=1)
    client_turn_id: str | None = None
    limit: int = Field(default=5, ge=1, le=20)
    validation_profile: str = "python_compile"
    show_debug_reasoning: bool = False

    @field_validator(
        "session_id", "repository_root", "message", "client_turn_id", "validation_profile"
    )
    @classmethod
    def strip_turn_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空")
        return stripped


class ChatTurnResponse(BaseModel):
    turn_id: str
    session_id: str
    run_id: str
    task_type: str
    status: str
    user_message: str
    assistant_message: str | None
    result: dict[str, object] | None
    error: str | None
    reasoning_available: bool
    created_at_epoch_ms: int
    updated_at_epoch_ms: int
