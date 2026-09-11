import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from codeinsight.application.context_budget import (
    CONTEXT_LIFECYCLE_POLICY_VERSION,
    ContextLifecyclePolicy,
)
from codeinsight.domain.errors import ModelConfigurationError
from codeinsight.domain.trace import CONTEXT_OVERFLOW
from codeinsight.infrastructure.event_log import InMemoryEventLog
from codeinsight.infrastructure.gateway_errors import (
    AuthenticationGatewayError,
    BackpressureError,
    BudgetExceededError,
    ContextOverflowGatewayError,
    InvalidResponseGatewayError,
    RateLimitGatewayError,
    TimeoutGatewayError,
    UpstreamGatewayError,
)
from codeinsight.infrastructure.model_gateway import (
    AdmissionController,
    CacheContext,
    CircuitBreaker,
    GatewayChatModel,
    GatewayRequest,
    InMemoryBudgetLedger,
    ModelGateway,
    RouteBudget,
    TokenBucketRateLimiter,
    configured_max_output_tokens,
    configured_routes_from_environment,
    default_gateway_from_environment,
    estimate_messages_tokens,
    resolve_route_budget,
)
from codeinsight.infrastructure.model_profiles import default_model_registry, default_routes
from codeinsight.infrastructure.provider_adapters import (
    FakeProviderAdapter,
    ProviderResponse,
)


def _request(**overrides: object) -> GatewayRequest:
    values: dict[str, object] = {
        "request_id": "req-1",
        "tenant_id": "default",
        "user_id": "local",
        "scene": "change-plan",
        "prompt_version": "planner-v1",
        "messages": ({"role": "user", "content": "inspect the repository"},),
        "estimated_input_tokens": 100,
        "reserved_output_tokens": 50,
        "required_capabilities": frozenset({"text", "tools", "structured_output"}),
        "tools": ({"type": "function", "function": {"name": "search_repository"}},),
    }
    values.update(overrides)
    return GatewayRequest(**values)  # type: ignore[arg-type]


def _success(model: str, content: str = '{"ok":true}') -> ProviderResponse:
    return ProviderResponse(content, (), model, 80, 20, 0, "stop")


def test_route_budget_shares_one_window_and_one_output_cap(monkeypatch) -> None:
    monkeypatch.setenv("CODEINSIGHT_MAX_OUTPUT_TOKENS", "40960")

    explain = resolve_route_budget("explain")
    tool_loop = resolve_route_budget("change-plan")

    assert explain.context_window_tokens == 128_000
    assert tool_loop.context_window_tokens == 128_000
    assert explain.reserved_output_tokens == 40_960
    assert tool_loop.reserved_output_tokens == 40_960
    assert explain.input_allowance == 128_000 - 40_960
    assert explain.policy_version == CONTEXT_LIFECYCLE_POLICY_VERSION


def test_route_budget_refuses_a_reservation_that_fills_the_window() -> None:
    with pytest.raises(ValueError):
        RouteBudget("explain", 100, 100, CONTEXT_LIFECYCLE_POLICY_VERSION)
    with pytest.raises(ValueError):
        RouteBudget("  ", 100, 10, CONTEXT_LIFECYCLE_POLICY_VERSION)


def test_messages_estimate_covers_roles_tools_and_json_escaping() -> None:
    single = ({"role": "user", "content": "inspect the repository"},)
    with_tools = (
        {"role": "user", "content": "inspect the repository"},
        {"role": "tool", "content": '{"path": "src/a.py", "line": 1}'},
    )

    assert estimate_messages_tokens(single) > 0
    assert estimate_messages_tokens(with_tools) > estimate_messages_tokens(single)


def test_overflow_guard_rejects_before_any_provider_call() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    with pytest.raises(ContextOverflowGatewayError, match="超出工作上下文窗口"):
        gateway.complete(
            _request(
                context_window_tokens=100,
                estimated_input_tokens=80,
                reserved_output_tokens=50,
            )
        )

    assert provider.called_models == []


def test_overflow_guard_reports_the_numbers_without_the_request_body() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    event_log = InMemoryEventLog()
    gateway = ModelGateway(provider=provider, event_log=event_log)

    with pytest.raises(ContextOverflowGatewayError):
        gateway.complete(
            _request(
                run_id="run-overflow",
                event_log=event_log,
                context_window_tokens=100,
                estimated_input_tokens=80,
                reserved_output_tokens=50,
            )
        )

    events = event_log.read_events("run-overflow")
    overflow = [event for event in events if event.event_type == CONTEXT_OVERFLOW]
    assert len(overflow) == 1
    payload = overflow[0].payload
    assert payload["context_window_tokens"] == "100"
    assert payload["estimated_input_tokens"] == "80"
    assert payload["reserved_output_tokens"] == "50"
    assert payload["overflow_tokens"] == "30"
    assert payload["guard_result"] == "overflow"
    assert "inspect the repository" not in json.dumps(payload, ensure_ascii=False)


def test_overflow_guard_lets_a_request_that_fits_through() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    response = gateway.complete(
        _request(
            context_window_tokens=100,
            estimated_input_tokens=50,
            reserved_output_tokens=50,
        )
    )

    assert response.content == '{"ok":true}'
    assert provider.called_models == ["deepseek-v4-flash"]


def test_request_without_a_declared_window_is_not_guarded() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)
    request = _request()

    assert request.overflow_guard_result == "not_checked"
    gateway.complete(request)

    assert provider.called_models == ["deepseek-v4-flash"]


def test_gateway_chat_model_estimates_from_the_serialized_messages() -> None:
    class ExpandingAssembler:
        def assemble(self, request):
            return SimpleNamespace(
                user_text="expanded context block " * 200,
                fitted=SimpleNamespace(estimate=SimpleNamespace(input_tokens=1)),
            )

    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    completion = GatewayChatModel(
        ModelGateway(provider=provider), context_assembler=ExpandingAssembler()
    ).complete("system-v1", "hello")

    # 估算必须跟着最终要发出去的 messages，而不是装配前的粗略字符串。
    assert completion.estimated_input_tokens is not None
    assert completion.estimated_input_tokens > 500


def test_gateway_chat_model_declares_the_shared_working_window() -> None:
    seen: list[GatewayRequest] = []

    class RecordingGateway:
        def complete(self, request: GatewayRequest) -> object:
            seen.append(request)
            return SimpleNamespace(
                content="{}",
                tool_calls=(),
                model="deepseek-v4-flash",
                input_tokens=1,
                output_tokens=1,
                reasoning_content=None,
            )

    model = GatewayChatModel(RecordingGateway())  # type: ignore[arg-type]
    model.complete("system-v1", "hello")
    model.complete_text("system-v1", "hello")
    model.complete_with_tools(({"role": "user", "content": "hi"},), ())

    assert [request.context_window_tokens for request in seen] == [128_000, 128_000, 128_000]
    assert [request.overflow_guard_result for request in seen] == ["fits", "fits", "fits"]


def test_structured_output_truncated_at_max_tokens_is_recognized() -> None:
    truncated = ProviderResponse("{", (), "deepseek-v4-flash", 80, 20, 0, "length")
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [truncated, truncated],
            "gpt-5.4-mini": [truncated, truncated],
            "gpt-5.4": [truncated, truncated],
        }
    )
    gateway = ModelGateway(provider=provider)

    with pytest.raises(InvalidResponseGatewayError):
        gateway.complete(_request(response_format={"type": "json_object"}))

    details = [
        event.payload.get("error_detail")
        for event in gateway.event_log.read_events("req-1")
        if event.event_type == "model_result"
    ]
    assert "structured_output_truncated" in details


def test_registry_uses_deepseek_as_best_and_cheapest_capable_fallback() -> None:
    registry = default_model_registry()
    routes = default_routes(registry)

    primary = registry.require("deepseek-v4-flash")
    fallback = registry.require(routes["change-plan"].fallback_model_ids[0])

    assert primary.input_price_per_million == 12
    assert primary.output_price_per_million == 36
    assert routes["change-plan"].primary_model_id == "deepseek-v4-flash"
    assert fallback.model_id == "gpt-5.4-mini"
    assert fallback.input_price_per_million == pytest.approx(0.75)
    assert {"tools", "structured_output"} <= fallback.capabilities


def test_environment_routes_disable_automatic_cross_endpoint_fallback(monkeypatch) -> None:
    monkeypatch.delenv("CODEINSIGHT_FALLBACK_MODELS", raising=False)

    routes = configured_routes_from_environment()

    assert all(route.fallback_model_ids == () for route in routes.values())


def test_environment_routes_require_explicit_registered_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CODEINSIGHT_FALLBACK_MODELS", "gpt-5.4-mini, gpt-5.4-mini")

    routes = configured_routes_from_environment()

    assert all(route.fallback_model_ids == ("gpt-5.4-mini",) for route in routes.values())


def test_environment_routes_reject_unknown_fallback_model(monkeypatch) -> None:
    monkeypatch.setenv("CODEINSIGHT_FALLBACK_MODELS", "model-that-is-not-registered")

    with pytest.raises(ModelConfigurationError, match="未登记模型"):
        configured_routes_from_environment()


def test_default_gateway_is_process_shared_for_circuit_and_cache(monkeypatch) -> None:
    default_gateway_from_environment.cache_clear()
    monkeypatch.setenv("CODEINSIGHT_GATEWAY_FAKE_MODE", "1")
    monkeypatch.delenv("CODEINSIGHT_FALLBACK_MODELS", raising=False)
    monkeypatch.setenv("CODEINSIGHT_API_KEY", "fake-key")
    monkeypatch.delenv("CODEINSIGHT_MYSQL_HOST", raising=False)
    monkeypatch.delenv("CODEINSIGHT_MYSQL_USER", raising=False)
    monkeypatch.delenv("CODEINSIGHT_MYSQL_DATABASE", raising=False)
    try:
        first = default_gateway_from_environment()
        second = default_gateway_from_environment()
        assert first is second
        assert all(route.fallback_model_ids == () for route in first.routes.values())
    finally:
        default_gateway_from_environment.cache_clear()


def test_output_budget_defaults_to_reasoning_safe_value_and_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv("CODEINSIGHT_MAX_OUTPUT_TOKENS", raising=False)
    assert configured_max_output_tokens() == 40_960

    monkeypatch.setenv("CODEINSIGHT_MAX_OUTPUT_TOKENS", "2048")
    assert configured_max_output_tokens() == 2_048

    monkeypatch.setenv("CODEINSIGHT_MAX_OUTPUT_TOKENS", "128")
    with pytest.raises(ModelConfigurationError, match="必须在"):
        configured_max_output_tokens()


def test_model_called_event_exposes_effective_output_budget(monkeypatch) -> None:
    monkeypatch.setenv("CODEINSIGHT_MAX_OUTPUT_TOKENS", "40960")
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    GatewayChatModel(gateway).complete("system-v1", "hello", run_id="run-budget")

    event = gateway.event_log.read_events("run-budget")[0]
    assert event.payload["max_output_tokens"] == "40960"


def test_gateway_chat_model_expands_context_budget_for_reasoning_and_answer(monkeypatch) -> None:
    monkeypatch.delenv("CODEINSIGHT_MAX_OUTPUT_TOKENS", raising=False)

    class RecordingAssembler:
        def __init__(self) -> None:
            self.request = None

        def assemble(self, request):
            self.request = request
            return SimpleNamespace(
                user_text="assembled prompt",
                fitted=SimpleNamespace(estimate=SimpleNamespace(input_tokens=12)),
            )

    assembler = RecordingAssembler()
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})

    GatewayChatModel(ModelGateway(provider=provider), context_assembler=assembler).complete(
        "system-v1", "hello"
    )

    # 工作窗口从生命周期策略取，不再由 Gateway 单独拼一个第二套默认值。
    assert assembler.request.max_tokens == ContextLifecyclePolicy().context_window_tokens
    assert assembler.request.reserved_output_tokens == 40_960
    assert assembler.request.policy_version == CONTEXT_LIFECYCLE_POLICY_VERSION


def test_gateway_chat_model_records_content_derived_prompt_version() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    completion = GatewayChatModel(gateway).complete("system-v1", "hello")

    assert completion.model == "deepseek-v4-flash"
    assert gateway.cost_records[0].prompt_version.startswith("prompt-sha256-")
    assert gateway.cost_records[0].prompt_version != "runtime-prompt"


def test_gateway_chat_model_text_route_does_not_request_json() -> None:
    class RecordingProvider:
        def __init__(self) -> None:
            self.request = None

        def invoke(self, request):
            self.request = request
            return _success(request.model, "你好！")

    provider = RecordingProvider()
    gateway = ModelGateway(provider=provider)  # type: ignore[arg-type]

    completion = GatewayChatModel(gateway).complete_text("general system", "你好")

    assert completion.content == "你好！"
    assert provider.request is not None
    assert provider.request.response_format is None
    assert gateway.cost_records[0].scene == "general-chat"
    assert gateway.cost_records[0].prompt_version == "general-chat-v1"


def test_gateway_chat_model_exposes_provider_reasoning_without_recording_it() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [
                ProviderResponse(
                    '{"ok":true}',
                    (),
                    "deepseek-v4-flash",
                    80,
                    20,
                    0,
                    "stop",
                    reasoning_content="provider debug reasoning",
                )
            ]
        }
    )
    gateway = ModelGateway(provider=provider)

    completion = GatewayChatModel(gateway).complete(
        "system-v1", "hello", run_id="run-reasoning"
    )

    assert completion.reasoning_content == "provider debug reasoning"
    events = gateway.event_log.read_events("run-reasoning")
    assert all("reasoning" not in str(event.payload) for event in events)


def test_rate_limit_retries_then_falls_back_and_records_price_version() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [
                RateLimitGatewayError("limited"),
                RateLimitGatewayError("limited again"),
            ],
            "gpt-5.4-mini": [_success("gpt-5.4-mini")],
        }
    )
    gateway = ModelGateway(provider=provider, max_retries=1)

    response = gateway.complete(_request())

    assert response.model == "gpt-5.4-mini"
    assert [item.model for item in response.attempts] == [
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        "gpt-5.4-mini",
    ]
    assert response.attempts[-1].fallback_reason == "RATE_LIMIT"
    assert response.cost.price_version == "frontier-stars-2026-09-04"
    assert response.cost.total_stars > 0
    assert len(gateway.cost_records) == 3
    assert [item.error_class for item in gateway.cost_records] == [
        "RATE_LIMIT",
        "RATE_LIMIT",
        None,
    ]


def test_authentication_error_never_falls_back() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [AuthenticationGatewayError("bad key")],
            "gpt-5.4-mini": [_success("gpt-5.4-mini")],
        }
    )

    with pytest.raises(AuthenticationGatewayError):
        ModelGateway(provider=provider).complete(_request())

    assert provider.called_models == ["deepseek-v4-flash"]


def test_budget_is_reserved_before_provider_call() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    ledger = InMemoryBudgetLedger(default_limit=100)

    with pytest.raises(BudgetExceededError):
        ModelGateway(provider=provider, budget_ledger=ledger).complete(_request())

    assert provider.called_models == []


def test_budget_reconcile_warns_when_actual_usage_is_much_higher() -> None:
    ledger = InMemoryBudgetLedger(default_limit=10_000)
    reservation = ledger.reserve("default", 100)

    ledger.reconcile(reservation, 200)

    assert ledger.underestimation_warnings == [(reservation.reservation_id, 100, 200)]


def test_route_tenant_token_bucket_refills() -> None:
    now = [10.0]
    limiter = TokenBucketRateLimiter(
        capacity=100, refill_tokens_per_second=10, clock=lambda: now[0]
    )
    limiter.consume("tenant", "explain", 100)
    with pytest.raises(BackpressureError):
        limiter.consume("tenant", "explain", 1)

    now[0] += 1
    limiter.consume("tenant", "explain", 10)


def test_invalid_json_gets_one_repair_attempt_then_cheaper_fallback() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [
                _success("deepseek-v4-flash", "not-json"),
                _success("deepseek-v4-flash", "still-not-json"),
            ],
            "gpt-5.4-mini": [_success("gpt-5.4-mini", json.dumps({"ok": True}))],
        }
    )
    request = _request(response_format={"type": "json_object"})

    response = ModelGateway(provider=provider, max_retries=0).complete(request)

    assert response.model == "gpt-5.4-mini"
    assert provider.called_models.count("deepseek-v4-flash") == 2


def test_gateway_accepts_fenced_structured_json_before_retrying_or_falling_back() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [
                _success(
                    "deepseek-v4-flash",
                    '```json\n{"ok":true}\n```',
                )
            ]
        }
    )
    request = _request(response_format={"type": "json_object"})

    response = ModelGateway(provider=provider, max_retries=0).complete(request)

    assert response.model == "deepseek-v4-flash"
    assert provider.called_models == ["deepseek-v4-flash"]


def test_invalid_structured_response_event_contains_safe_diagnostic() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [
                _success("deepseek-v4-flash", "not-json"),
                _success("deepseek-v4-flash", "still-not-json"),
            ]
        }
    )
    routes = default_routes(default_model_registry())
    routes["change-plan"] = replace(routes["change-plan"], fallback_model_ids=())
    request = _request(response_format={"type": "json_object"}, run_id="run-invalid-json")
    gateway = ModelGateway(provider=provider, routes=routes, max_retries=0)

    with pytest.raises(InvalidResponseGatewayError):
        gateway.complete(request)

    events = gateway.event_log.read_events("run-invalid-json")
    failed_results = [
        event
        for event in events
        if event.event_type == "model_result" and event.payload["outcome"] == "error"
    ]
    assert failed_results[-1].payload["error_detail"] == "structured_output_invalid_json"
    assert failed_results[-1].payload["finish_reason"] == "stop"


@pytest.mark.parametrize("error", [UpstreamGatewayError("500"), TimeoutGatewayError("slow")])
def test_retryable_upstream_failures_reach_cheaper_model(error: Exception) -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [error],
            "gpt-5.4-mini": [_success("gpt-5.4-mini")],
        }
    )

    response = ModelGateway(provider=provider, max_retries=0).complete(_request())

    assert response.model == "gpt-5.4-mini"


def test_circuit_opens_after_repeated_primary_failures() -> None:
    provider = FakeProviderAdapter(
        {
            "deepseek-v4-flash": [
                UpstreamGatewayError("500-a"),
                UpstreamGatewayError("500-b"),
            ],
            "gpt-5.4-mini": [
                _success("gpt-5.4-mini"),
                _success("gpt-5.4-mini"),
                _success("gpt-5.4-mini"),
            ],
        }
    )
    gateway = ModelGateway(
        provider=provider,
        max_retries=0,
        circuit_breaker=CircuitBreaker(failure_threshold=2, cooldown_seconds=60),
    )

    for number in range(3):
        gateway.complete(_request(request_id=f"req-circuit-{number}"))

    assert provider.called_models.count("deepseek-v4-flash") == 2
    assert provider.called_models.count("gpt-5.4-mini") == 3


def test_backpressure_rejects_before_provider_call() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    admission = AdmissionController(max_concurrency=1)
    admission.acquire()
    gateway = ModelGateway(provider=provider, admission=admission)

    with pytest.raises(BackpressureError):
        gateway.complete(_request())

    admission.release()
    assert provider.called_models == []


def test_gateway_emits_public_model_events_without_prompts() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    gateway.complete(_request(run_id="run-events"))
    events = gateway.event_log.read_events("run-events")

    assert [event.event_type for event in events] == ["model_called", "model_result"]
    assert all("messages" not in event.payload for event in events)
    assert events[-1].payload["model"] == "deepseek-v4-flash"


def test_price_version_keeps_historical_cost_stable() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    response = gateway.complete(_request())
    recorded = response.cost

    assert recorded.price_version == "frontier-stars-2026-09-04"
    assert response.cost == recorded


def test_semantic_cache_is_repo_and_index_scoped_and_skips_workspace_state() -> None:
    provider = FakeProviderAdapter(
        {"deepseek-v4-flash": [_success("deepseek-v4-flash"), _success("deepseek-v4-flash")]}
    )
    gateway = ModelGateway(provider=provider)
    base = CacheContext("repo", "fp-a", "idx-1", False, False)

    first = gateway.complete(
        _request(
            scene="explain",
            required_capabilities=frozenset({"text"}),
            tools=(),
            cache_context=base,
        )
    )
    second = gateway.complete(
        _request(
            scene="explain",
            required_capabilities=frozenset({"text"}),
            tools=(),
            cache_context=base,
        )
    )
    other_repo = gateway.complete(
        _request(
            scene="explain",
            required_capabilities=frozenset({"text"}),
            tools=(),
            cache_context=CacheContext("repo", "fp-b", "idx-1", False, False),
        )
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert other_repo.cache_hit is False
    assert provider.called_models == ["deepseek-v4-flash", "deepseek-v4-flash"]

    live = _request(
        request_id="req-live",
        scene="explain",
        required_capabilities=frozenset({"text"}),
        tools=(),
        cache_context=CacheContext("repo", "fp-a", "idx-1", True, False),
    )
    assert gateway.semantic_cache.key_for(live, "deepseek-v4-flash") is None
