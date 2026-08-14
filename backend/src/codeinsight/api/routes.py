"""由 CodeInsight 应用用例提供支持的版本化 HTTP 路由。"""

from collections.abc import Callable
from dataclasses import replace

from fastapi import APIRouter, HTTPException

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.api.schemas import (
    AgentAnswerRequest,
    AgentAnswerResponse,
    AgentEventResponse,
    AnswerRequest,
    AnswerResponse,
    AutoAnswerRequest,
    AutoAnswerResponse,
    AutoEventResponse,
    AutoSubQuestionResponse,
    CitationResponse,
    HealthResponse,
    QueryPlanResponse,
    QueryPlanSubQuestionResponse,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
    TokenUsageResponse,
)
from codeinsight.application.agent_answer_repository import agent_answer_repository
from codeinsight.application.answer_repository import answer_repository
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.query_router import QueryRouterResult, route_question
from codeinsight.application.search_repository import search_repository
from codeinsight.domain.answer import AutoAnswer, AutoAnswerEvent, SubQuestionAnswer
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel

ModelFactory = Callable[[], OpenAIChatModel]
EmbeddingFactory = Callable[[], OpenAIEmbeddingModel]


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _citation_responses(citations):
    return [
        CitationResponse(
            evidence_id=citation.evidence_id,
            relative_path=citation.relative_path,
            start_line=citation.start_line,
            end_line=citation.end_line,
        )
        for citation in citations
    ]


def _auto_from_agent(router_result: QueryRouterResult, agent_result) -> AutoAnswer:
    result = agent_result.result
    subquestions = agent_result.subquestions
    if not subquestions:
        subquestions = tuple(
            SubQuestionAnswer(
                question=item.question,
                intent=item.intent,
                retrieval_mode=item.retrieval_mode,
                outcome=result.outcome,
                answer=result.answer,
                citations=result.citations,
            )
            for item in router_result.plan.subquestions
        )
    events = tuple(
        AutoAnswerEvent(event.sequence, event.step, event.summary) for event in agent_result.events
    )
    return AutoAnswer(
        outcome=result.outcome,
        answer=result.answer,
        citations=result.citations,
        retrieval_mode="auto",
        model=result.model,
        prompt_version=result.prompt_version,
        input_tokens=agent_result.input_tokens,
        output_tokens=agent_result.output_tokens,
        plan=router_result.plan,
        subquestions=subquestions,
        router_model=router_result.model,
        router_input_tokens=router_result.input_tokens,
        router_output_tokens=router_result.output_tokens,
        router_elapsed_milliseconds=router_result.elapsed_milliseconds,
        embedding_input_tokens=agent_result.embedding_input_tokens,
        fallback_reason=router_result.fallback_reason,
        events=events,
    )


def _auto_response(result: AutoAnswer) -> AutoAnswerResponse:
    plan = result.plan.to_dict()
    return AutoAnswerResponse(
        outcome=result.outcome,
        answer=result.answer,
        citations=_citation_responses(result.citations),
        retrieval_mode=result.retrieval_mode,
        model=result.model,
        prompt_version=result.prompt_version,
        usage=TokenUsageResponse(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
        plan=QueryPlanResponse(
            original_question=plan["original_question"],
            language=plan["language"],
            normalized_question=plan["normalized_question"],
            subquestions=[QueryPlanSubQuestionResponse(**item) for item in plan["subquestions"]],
            retrieval_modes=plan["retrieval_modes"],
            execution_route=plan["execution_route"],
            confidence=plan["confidence"],
            fallback_reason=plan["fallback_reason"],
        ),
        subquestions=[
            AutoSubQuestionResponse(
                question=item.question,
                intent=item.intent,
                retrieval_mode=item.retrieval_mode,
                outcome=item.outcome,
                answer=item.answer,
                citations=_citation_responses(item.citations),
            )
            for item in result.subquestions
        ],
        router_model=result.router_model,
        router_usage=TokenUsageResponse(
            input_tokens=result.router_input_tokens,
            output_tokens=result.router_output_tokens,
        ),
        router_elapsed_milliseconds=result.router_elapsed_milliseconds,
        embedding_input_tokens=result.embedding_input_tokens,
        fallback_reason=result.fallback_reason,
        events=[
            AutoEventResponse(
                sequence=event.sequence,
                step=event.step,
                summary=event.summary,
            )
            for event in result.events
        ],
    )


def create_router(
    model_factory: ModelFactory,
    embedding_factory: EmbeddingFactory = OpenAIEmbeddingModel.from_environment,
) -> APIRouter:
    """创建带有可注入模型组合边界的 API Router。"""
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    # 已停用的产品入口：保留实现以便将来明确恢复，但不注册 POST /api/v1/search。
    # @router.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            embedding_model = embedding_factory() if request.retrieval_mode == "hybrid" else None
            results = search_repository(
                request.repository_root,
                request.question,
                limit=request.limit,
                retrieval_mode=request.retrieval_mode,
                semantic_embed=embedding_model.embed if embedding_model else None,
            )
        except ModelConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return SearchResponse(
            retrieval_mode=request.retrieval_mode,
            results=[
                SearchHitResponse(
                    rank=item.rank,
                    score=item.score,
                    relative_path=item.chunk.relative_path,
                    start_line=item.chunk.start_line,
                    end_line=item.chunk.end_line,
                    excerpt=_first_line(item.chunk.text),
                    symbol_path=item.chunk.symbol_path,
                    retrieval_reason=item.retrieval_reason,
                )
                for item in results
            ],
        )

    # 已停用的产品入口：Auto Answer 内部仍复用 answer_repository。
    # @router.post("/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        try:
            model = model_factory()
            embedding_model = embedding_factory() if request.retrieval_mode == "hybrid" else None
            result = answer_repository(
                request.repository_root,
                request.question,
                generate=model.generate,
                limit=request.limit,
                retrieval_mode=request.retrieval_mode,
                semantic_embed=embedding_model.embed if embedding_model else None,
            )
        except ModelConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ModelCallError, ModelResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return AnswerResponse(
            outcome=result.outcome,
            answer=result.answer,
            citations=[
                CitationResponse(
                    evidence_id=citation.evidence_id,
                    relative_path=citation.relative_path,
                    start_line=citation.start_line,
                    end_line=citation.end_line,
                )
                for citation in result.citations
            ],
            retrieval_mode=result.retrieval_mode,
            model=result.model,
            prompt_version=result.prompt_version,
            usage=TokenUsageResponse(
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
            ),
        )

    # 已停用的产品入口：Auto Answer 内部仍复用 Agent 工作流。
    # @router.post("/agent/answer", response_model=AgentAnswerResponse)
    def agent_answer(request: AgentAnswerRequest) -> AgentAnswerResponse:
        try:
            model = model_factory()
            embedding_model = embedding_factory() if request.retrieval_mode == "hybrid" else None
            agent_result = agent_answer_repository(
                request.repository_root,
                request.question,
                complete=model.complete,
                limit=request.limit,
                retrieval_mode=request.retrieval_mode,
                semantic_embed=embedding_model.embed if embedding_model else None,
            )
        except ModelConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ModelCallError, ModelResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        result = agent_result.result
        return AgentAnswerResponse(
            outcome=result.outcome,
            answer=result.answer,
            citations=[
                CitationResponse(
                    evidence_id=citation.evidence_id,
                    relative_path=citation.relative_path,
                    start_line=citation.start_line,
                    end_line=citation.end_line,
                )
                for citation in result.citations
            ],
            retrieval_mode=result.retrieval_mode,
            model=result.model,
            prompt_version=result.prompt_version,
            usage=TokenUsageResponse(
                input_tokens=agent_result.input_tokens,
                output_tokens=agent_result.output_tokens,
            ),
            revisions=agent_result.revisions,
            events=[
                AgentEventResponse(
                    sequence=event.sequence,
                    step=event.step,
                    summary=event.summary,
                )
                for event in agent_result.events
            ],
        )

    @router.post("/auto/answer", response_model=AutoAnswerResponse)
    def auto_answer(request: AutoAnswerRequest) -> AutoAnswerResponse:
        try:
            model = model_factory()
            router_result = route_question(request.question, complete=model.complete)
            if request.force_route:
                plan = replace(
                    router_result.plan,
                    execution_route=request.force_route,
                    fallback_reason="forced_route",
                )
                router_result = replace(
                    router_result,
                    plan=plan,
                    used_fallback=False,
                    fallback_reason="forced_route",
                )
            embedding_model = (
                embedding_factory()
                if router_result.plan.execution_route != "insufficient"
                else None
            )
            if router_result.plan.execution_route == "agent":
                agent_result = run_citation_agent(
                    request.repository_root,
                    request.question,
                    complete=model.complete,
                    limit=request.limit,
                    retrieval_mode="auto",
                    semantic_embed=embedding_model.embed if embedding_model else None,
                    query_plan=router_result.plan,
                )
                result = _auto_from_agent(router_result, agent_result)
            else:
                result = auto_answer_repository(
                    request.repository_root,
                    router_result=router_result,
                    generate=model.generate,
                    limit=request.limit,
                    semantic_embed=embedding_model.embed if embedding_model else None,
                )
        except ModelConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ModelCallError, ModelResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _auto_response(result)

    return router
