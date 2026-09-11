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
from codeinsight.domain.chat import (
    CHAT_MANUAL_REQUIRED,
    CHAT_UNKNOWN,
    CHAT_WAITING_APPROVAL,
)
from codeinsight.domain.trace import RUN_FINISHED, STATE_TRANSITIONED
from codeinsight.infrastructure.chat_runtime import ChatRuntimeError

# 事件流在这里把控制权交回客户端：等审批、以及两种「停下来等人」的状态。
# 等校验不在其中——它没有用户决策点，后续 continuation 会继续往同一个 run 上写。
_STOP_STREAM_STATUSES = frozenset(
    {CHAT_WAITING_APPROVAL, CHAT_UNKNOWN, CHAT_MANUAL_REQUIRED}
)


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
        except ChatDispatchError as error:
            # 审批本身已经登记过，但没有任何 Worker 接手这次续跑。返回 503 而
            # 不是 2xx：不能给用户一个「已经批准并开始了」的姿态。
            raise HTTPException(status_code=503, detail=str(error)) from error
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
        run_id = conversation_service.resolve_run_id(turn_id)
        if run_id is None:
            raise HTTPException(status_code=404, detail="聊天 Run 不存在")

        def stream() -> Iterator[str]:
            # 回放与订阅在一步里完成：已落事实的事件从 Store 取，新事件从广播收，
            # 两边用 sequence 去重。断线重连因此既不会重复，也不会漏。
            subscription = conversation_service.runtime.subscribe(
                run_id, after_sequence=after_sequence
            )
            subscriber = subscription.subscriber
            last_sequence = after_sequence
            try:
                finished = False
                for event in subscription.backlog:
                    last_sequence = max(last_sequence, event.sequence)
                    yield _event_frame(event)
                    if _ends_the_stream(event):
                        finished = True
                        break
                for payload in subscription.debug_backlog:
                    yield _debug_frame(payload)
                if finished or conversation_service.run_is_settled(run_id):
                    return
                while True:
                    try:
                        kind, payload = subscriber.get(timeout=8)
                    except Empty:
                        yield ": keep-alive\n\n"
                        if conversation_service.run_is_settled(run_id):
                            return
                        continue
                    if kind == "debug_reasoning":
                        yield _debug_frame(payload)
                        continue
                    event = payload
                    if event.sequence <= last_sequence:
                        # 广播与回放重叠的部分只推一次。
                        continue
                    last_sequence = event.sequence
                    yield _event_frame(event)
                    if _ends_the_stream(event):
                        return
            finally:
                conversation_service.runtime.unsubscribe(run_id, subscriber)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router



def _ends_the_stream(event) -> bool:
    """什么事件表示「这条连接可以收尾了」。

    等审批要停下来等用户动作（客户端随后重新订阅）；run_finished 是本轮结束。
    等校验不停：它没有用户决策点，后续 continuation 会继续往同一个 run 上写。
    """

    if event.event_type == RUN_FINISHED:
        return True
    return (
        event.event_type == STATE_TRANSITIONED
        and event.payload.get("to_status") in _STOP_STREAM_STATUSES
    )

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
