"""Q-011 Task 10 验收：从 HTTP 受理到校验结束，关联 ID 一路不丢。

一次 change 会跨过四个边界：HTTP API → AgentRunWorker → 模型/工具 → ValidationWorker。
这里不查日志，而是逐个查事实：Run 记录、事件流、变更结果、校验任务、Span 都必须能
用同一个 run_id 串起来；续跑换了 attempt，却仍是同一个 run。
"""

from __future__ import annotations

from pathlib import Path

from codeinsight.infrastructure.model_gateway import ModelGateway
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.provider_adapters import FakeProviderAdapter
from tests.integration.test_chat_change_async import (
    _build,
    _session,
    _submit_change,
    _wait_for,
)
from tests.unit.infrastructure.test_model_gateway import (
    DEFAULT_MODEL_ID,
    _request,
    _success,
)


def test_one_change_run_keeps_its_identity_across_every_boundary(tmp_path: Path) -> None:
    client, repo, _base, _model = _build(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL", preview.get("error")
    approved = client.post(f"/api/v2/chat/turns/{preview['turn_id']}/approve")
    assert approved.status_code == 202, approved.text
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    events = conversation.runtime.event_log.read_events(record.run_id)

    # 1) Worker 边界：三次尝试，同一个 run，三个新 task_id。
    claimed = [event for event in events if event.event_type == "worker_claimed"]
    assert [event.payload["attempt"] for event in claimed] == ["1", "2", "3"]
    assert [event.payload["task_kind"] for event in claimed] == [
        "agent_run",
        "resume_after_approval",
        "resume_after_validation",
    ]
    assert len({event.payload["task_id"] for event in claimed}) == 3
    assert all(event.run_id == record.run_id for event in events)

    # 2) 变更事实与校验任务边界：都挂在同一个 run 上。
    result = conversation.change_service.get_result(record.run_id, record.patch_id)
    assert result is not None
    assert result.status == "COMPLETED"
    validation_tasks = [
        task
        for task in conversation.validation_queue.list_tasks()
        if task.run_id == record.run_id
    ]
    assert len(validation_tasks) == 1
    assert validation_tasks[0].payload["patch_id"] == record.patch_id

    # 3) Span 边界：Worker 执行留下的是同一个 run、同一个 turn、三次不同尝试。
    worker_spans = [
        span
        for span in get_telemetry().records()
        if span.component == "worker"
        and span.operation == "agent_run"
        and span.attributes.get("run_id") == record.run_id
    ]
    assert [span.attributes["attempt_id"] for span in worker_spans] == ["1", "2", "3"]
    assert {span.attributes["turn_id"] for span in worker_spans} == {preview["turn_id"]}

    # 4) 指标边界：这一次尝试进了低基数指标，而标识没有变成 label。
    rendered = get_telemetry().metrics().decode("utf-8")
    assert "codeinsight_agent_run_attempts_total" in rendered
    assert 'task_kind="resume_after_validation"' in rendered
    assert record.run_id not in rendered
    assert preview["turn_id"] not in rendered


def test_the_gateway_span_carries_the_run_it_serves() -> None:
    """模型调用也要能回答「这次调用属于哪一轮」，而不是只有一个本地 request_id。"""

    provider = FakeProviderAdapter({DEFAULT_MODEL_ID: [_success(DEFAULT_MODEL_ID)]})
    gateway = ModelGateway(provider=provider)

    gateway.complete(_request(run_id="run-trace-10"))

    gateway_spans = [
        span
        for span in get_telemetry().records()
        if span.component == "gateway" and span.operation == "model_attempt"
    ]
    assert gateway_spans
    latest = gateway_spans[-1]
    assert latest.attributes["run_id"] == "run-trace-10"
    # 每次调用还有自己的 attempt_id：同一轮里的多次调用必须能分开看。
    assert latest.attributes["attempt_id"]
