"""装配层：投递通道按环境选择，且显式注入永远优先。

这条规则有两面，缺一面都会出错：本机没配 broker 时切到跨进程，会让每一轮永远停在
QUEUED（没有 Worker 进程在跑）；而忽略了调用方显式传入的通道，Worker 进程就会用
自己的 broker 通道去覆盖部署时的选择。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from codeinsight.api.app import ENV_CELERY_BROKER_URL, create_app
from codeinsight.application.agent_run_dispatcher import (
    AgentRunDispatcher,
    CallbackAgentRunTransport,
    InMemoryAgentRunTransport,
)
from codeinsight.infrastructure.celery_agent_dispatcher import (
    CeleryAgentRunTransport,
    CeleryValidationTransport,
)


def _dispatcher(app: FastAPI) -> AgentRunDispatcher:
    return app.state.codeinsight_conversation.agent_run_dispatcher


def _transport(app: FastAPI) -> object:
    # Dispatcher 是应用层唯一投递入口，没有公开的 transport 读取口；装配用例
    # 要断言的正是它内部选了哪一条通道。
    return _dispatcher(app)._transport


def test_without_broker_the_process_internal_callback_is_kept() -> None:
    """没配 broker：默认行为不变，仍然是进程内回调。"""

    app = create_app()

    assert isinstance(_transport(app), CallbackAgentRunTransport)


def test_broker_environment_switches_both_transports(monkeypatch: pytest.MonkeyPatch) -> None:
    """配了 broker：两条通道一起换成 Celery，不能只换一条。"""

    monkeypatch.setenv(ENV_CELERY_BROKER_URL, "redis://127.0.0.1:6380/1")

    app = create_app()
    service = app.state.codeinsight_conversation

    assert isinstance(_transport(app), CeleryAgentRunTransport)
    assert isinstance(service.validation_coordinator._transport, CeleryValidationTransport)


def test_blank_broker_value_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """空字符串不是「配了 broker」：空值当成没配，否则一个空变量就能关掉执行。"""

    monkeypatch.setenv(ENV_CELERY_BROKER_URL, "   ")

    app = create_app()

    assert isinstance(_transport(app), CallbackAgentRunTransport)


def test_explicit_transport_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式注入优先：Worker 进程就是靠这一点用自己的 broker 通道。"""

    monkeypatch.setenv(ENV_CELERY_BROKER_URL, "redis://127.0.0.1:6380/1")
    injected = InMemoryAgentRunTransport()

    app = create_app(agent_run_transport=injected)

    assert _transport(app) is injected
