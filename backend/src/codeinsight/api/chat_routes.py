"""统一对话 HTTP/SSE 入口。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from queue import Empty

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from codeinsight.api.schemas import (
    ChatSessionRequest,
    ChatSessionResponse,
    ChatTurnRequest,
    ChatTurnResponse,
)
from codeinsight.application.conversation_service import (
    ChatDispatchError,
    ConversationService,
)
from codeinsight.infrastructure.chat_runtime import ChatRuntimeError


def create_chat_router(conversation_service: ConversationService) -> APIRouter:
    router = APIRouter(prefix="/api/v2/chat")

    @router.post("/sessions", response_model=ChatSessionResponse)
    def create_session(request: ChatSessionRequest) -> ChatSessionResponse:
        try:
            payload = conversation_service.create_session(
                request.repository_root, session_id=request.session_id
            )
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ChatSessionResponse(**payload)

    @router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
    def get_session(
        session_id: str, repository_root: str = Query(min_length=1)
    ) -> ChatSessionResponse:
        try:
            payload = conversation_service.get_session(session_id, repository_root)
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ChatSessionResponse(**payload)

    @router.post("/turns", response_model=ChatTurnResponse, status_code=202)
    def submit_turn(request: ChatTurnRequest) -> ChatTurnResponse:
        try:
            session_id = request.session_id
            if session_id is None:
                session_id = str(
                    conversation_service.create_session(request.repository_root)["session_id"]
                )
            turn = conversation_service.submit_turn(
                session_id=session_id,
                repository_root=request.repository_root,
                message=request.message,
                client_turn_id=request.client_turn_id,
                limit=request.limit,
                validation_profile=request.validation_profile,
                show_debug_reasoning=request.show_debug_reasoning,
            )
        except ChatRuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ChatDispatchError as error:
            # 受理失败：Run 已经留下失败事实，客户端应当知道这一轮没被受理，
            # 而不是拿着一个永远不会有人执行的 202 等下去。
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return ChatTurnResponse(**turn.as_dict())

    @router.get("/turns/{turn_id}", response_model=ChatTurnResponse)
    def get_turn(turn_id: str) -> ChatTurnResponse:
        try:
            turn = conversation_service.runtime.get_turn(turn_id)
        except ChatRuntimeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return ChatTurnResponse(**turn.as_dict())

    @router.get("/runs/{run_id}", response_model=ChatTurnResponse)
    def get_run(run_id: str) -> ChatTurnResponse:
        try:
            turn = conversation_service.runtime.get_turn_by_run_id(run_id)
        except ChatRuntimeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return ChatTurnResponse(**turn.as_dict())

    @router.post("/turns/{turn_id}/approve", response_model=ChatTurnResponse, status_code=202)
    def approve_turn(turn_id: str) -> ChatTurnResponse:
        try:
            turn = conversation_service.approve_turn(turn_id)
        except ChatRuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return ChatTurnResponse(**turn.as_dict())

    @router.post("/turns/{turn_id}/cancel", response_model=ChatTurnResponse)
    def cancel_turn(turn_id: str) -> ChatTurnResponse:
        try:
            turn = conversation_service.cancel_turn(turn_id)
        except ChatRuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return ChatTurnResponse(**turn.as_dict())

    @router.get("/turns/{turn_id}/events")
    def turn_events(
        turn_id: str, after_sequence: int = Query(default=0, ge=0)
    ) -> StreamingResponse:
        try:
            turn = conversation_service.runtime.get_turn(turn_id)
        except ChatRuntimeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        def stream() -> Iterator[str]:
            backlog, debug_backlog, subscriber = conversation_service.runtime.subscribe(
                turn.run_id, after_sequence=after_sequence
            )
            last_sequence = after_sequence
            try:
                stop_after_backlog = False
                for event in backlog:
                    last_sequence = max(last_sequence, event.sequence)
                    yield _event_frame(event)
                    if event.event_type == "run_finished" or (
                        event.event_type == "state_transitioned"
                        and event.payload.get("to_status") == "WAITING_APPROVAL"
                    ):
                        stop_after_backlog = True
                        break
                for payload in debug_backlog:
                    yield _debug_frame(payload)
                if stop_after_backlog or conversation_service.runtime.is_terminal(turn.run_id):
                    return
                while True:
                    try:
                        kind, payload = subscriber.get(timeout=8)
                    except Empty:
                        yield ": keep-alive\n\n"
                        if conversation_service.runtime.is_terminal(turn.run_id):
                            return
                        continue
                    if kind == "debug_reasoning":
                        yield _debug_frame(payload)
                        continue
                    event = payload
                    if event.sequence <= last_sequence:
                        continue
                    last_sequence = event.sequence
                    yield _event_frame(event)
                    if event.event_type == "run_finished" or (
                        event.event_type == "state_transitioned"
                        and event.payload.get("to_status") == "WAITING_APPROVAL"
                    ):
                        return
            finally:
                conversation_service.runtime.unsubscribe(turn.run_id, subscriber)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _event_frame(event) -> str:
    payload = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "occurred_at_epoch_ms": event.occurred_at_epoch_ms,
        "payload": event.payload,
    }
    data = json.dumps(payload, ensure_ascii=False)
    return f"id: {event.sse_event_id}\nevent: {event.event_type}\ndata: {data}\n\n"


def _debug_frame(payload: dict[str, object]) -> str:
    return f"event: debug_reasoning\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
