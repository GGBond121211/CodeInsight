"""由 CodeInsight 应用用例提供支持的版本化 HTTP 路由。"""

from collections.abc import Callable
from dataclasses import replace

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.api.schemas import (
    AutoAnswerRequest,
    AutoAnswerResponse,
    AutoEventResponse,
    AutoSubQuestionResponse,
    ChangeApplyRequest,
    ChangeApproveRequest,
    ChangeApproveResponse,
    ChangeCancelResponse,
    ChangeEventResponse,
    ChangePreviewRequest,
    ChangePreviewResponse,
    ChangeResultResponse,
    ChangeRollbackRequest,
    CitationResponse,
    HealthResponse,
    QueryPlanResponse,
    QueryPlanSubQuestionResponse,
    TokenUsageResponse,
)
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.change_service import ChangeRequestError, ChangeService
from codeinsight.application.query_router import QueryRouterResult, route_question
from codeinsight.domain.answer import AutoAnswer, AutoAnswerEvent, SubQuestionAnswer
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel
from codeinsight.infrastructure.reranker import OpenAITextReranker, Reranker

ModelFactory = Callable[[], OpenAIChatModel]
EmbeddingFactory = Callable[[], OpenAIEmbeddingModel]
RerankerFactory = Callable[[], Reranker]


def _citation_responses(citations):
    responses = []
    for citation in citations:
        responses.append(
            CitationResponse(
                evidence_id=citation.evidence_id,
                relative_path=citation.relative_path,
                start_line=citation.start_line,
                end_line=citation.end_line,
            )
        )
    return responses


def _auto_from_agent(router_result: QueryRouterResult, agent_result) -> AutoAnswer:
    result = agent_result.result
    subquestions = agent_result.subquestions
    if not subquestions:
        fallback_subquestions: list[SubQuestionAnswer] = []
        for item in router_result.plan.subquestions:
            fallback_subquestions.append(
                SubQuestionAnswer(
                    question=item.question,
                    intent=item.intent,
                    retrieval_mode=item.retrieval_mode,
                    outcome=result.outcome,
                    answer=result.answer,
                    citations=result.citations,
                )
            )
        subquestions = tuple(fallback_subquestions)
    events_list: list[AutoAnswerEvent] = []
    for event in agent_result.events:
        events_list.append(
            AutoAnswerEvent(event.sequence, event.step, event.summary)
        )
    events = tuple(events_list)
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
    plan_subquestions: list[QueryPlanSubQuestionResponse] = []
    for item in plan["subquestions"]:
        plan_subquestions.append(QueryPlanSubQuestionResponse(**item))

    subquestion_responses: list[AutoSubQuestionResponse] = []
    for item in result.subquestions:
        subquestion_responses.append(
            AutoSubQuestionResponse(
                question=item.question,
                intent=item.intent,
                retrieval_mode=item.retrieval_mode,
                outcome=item.outcome,
                answer=item.answer,
                citations=_citation_responses(item.citations),
            )
        )

    event_responses: list[AutoEventResponse] = []
    for event in result.events:
        event_responses.append(
            AutoEventResponse(
                sequence=event.sequence,
                step=event.step,
                summary=event.summary,
            )
        )

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
            subquestions=plan_subquestions,
            retrieval_modes=plan["retrieval_modes"],
            execution_route=plan["execution_route"],
            confidence=plan["confidence"],
            fallback_reason=plan["fallback_reason"],
        ),
        subquestions=subquestion_responses,
        router_model=result.router_model,
        router_usage=TokenUsageResponse(
            input_tokens=result.router_input_tokens,
            output_tokens=result.router_output_tokens,
        ),
        router_elapsed_milliseconds=result.router_elapsed_milliseconds,
        embedding_input_tokens=result.embedding_input_tokens,
        fallback_reason=result.fallback_reason,
        events=event_responses,
    )


def create_router(
    model_factory: ModelFactory,
    embedding_factory: EmbeddingFactory = OpenAIEmbeddingModel.from_environment,
    reranker_factory: RerankerFactory = OpenAITextReranker.from_environment,
) -> APIRouter:
    """创建带有可注入模型组合边界的 API Router。"""
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

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
            reranker = (
                reranker_factory()
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
                    reranker=reranker,
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
                    reranker=reranker,
                )
        except ModelConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ModelCallError, ModelResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _auto_response(result)

    return router


def create_change_router(change_service: ChangeService | None = None) -> APIRouter:
    """Step 6 变更 API；与只读的 v1 入口分开，避免改变旧契约。"""
    router = APIRouter(prefix="/api/v2")
    changes = change_service or ChangeService()

    @router.post("/change/preview", response_model=ChangePreviewResponse)
    def change_preview(request: ChangePreviewRequest) -> ChangePreviewResponse:
        try:
            result = changes.preview(
                request.repository_root,
                run_id=request.run_id,
                path=request.path,
                new_content=request.new_content,
                validation_profile=request.validation_profile,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (ChangeRequestError, ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ChangePreviewResponse(**result.as_dict())

    @router.post("/change/approve", response_model=ChangeApproveResponse)
    def change_approve(request: ChangeApproveRequest) -> ChangeApproveResponse:
        try:
            token = changes.approve(
                request.run_id,
                request.patch_id,
                expires_in_seconds=request.expires_in_seconds,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (ChangeRequestError, ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return ChangeApproveResponse(
            run_id=request.run_id,
            patch_id=request.patch_id,
            approval_token=token,
            expires_in_seconds=request.expires_in_seconds,
        )

    @router.post("/change/apply", response_model=ChangeResultResponse)
    def change_apply(request: ChangeApplyRequest) -> ChangeResultResponse:
        try:
            result = changes.apply(request.run_id, request.patch_id, request.approval_token)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ChangeRequestError, ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=503, detail="执行异常，请查询变更结果；未确认状态不得自动重放"
            ) from error
        return ChangeResultResponse(**result.as_dict())

    @router.post("/change/rollback", response_model=ChangeResultResponse)
    def change_rollback(request: ChangeRollbackRequest) -> ChangeResultResponse:
        try:
            result = changes.rollback(request.run_id, request.patch_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ChangeRequestError, ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=503, detail="回滚未确认完成，请查询变更结果并人工核对工作区"
            ) from error
        return ChangeResultResponse(**result.as_dict())

    @router.post("/change/{run_id}/cancel", response_model=ChangeCancelResponse)
    def change_cancel(run_id: str) -> ChangeCancelResponse:
        try:
            status, immediate = changes.cancel(run_id)
        except ChangeRequestError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return ChangeCancelResponse(status=status, run_id=run_id, immediate=immediate)

    @router.get("/change/{run_id}/patches/{patch_id}", response_model=ChangeResultResponse)
    def change_result(run_id: str, patch_id: str) -> ChangeResultResponse:
        try:
            result = changes.get_result(run_id, patch_id)
        except ChangeRequestError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if result is None:
            raise HTTPException(status_code=404, detail="变更结果尚未生成")
        return ChangeResultResponse(**result.as_dict())

    @router.get("/change/{run_id}/events")
    def change_events(
        run_id: str, after_sequence: int = Query(default=0, ge=0)
    ) -> StreamingResponse:
        events = changes.events_after(run_id, after_sequence)

        def stream():
            for event in events:
                payload = ChangeEventResponse(
                    event_id=event.event_id,
                    run_id=event.run_id,
                    sequence=event.sequence,
                    event_type=event.event_type,
                    occurred_at_epoch_ms=event.occurred_at_epoch_ms,
                    payload=event.payload,
                ).model_dump_json()
                yield f"id: {event.sse_event_id}\nevent: {event.event_type}\ndata: {payload}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return router
