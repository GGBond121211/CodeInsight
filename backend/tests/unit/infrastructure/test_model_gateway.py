import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from codeinsight.domain.errors import ModelConfigurationError
from codeinsight.infrastructure.gateway_errors import (
    AuthenticationGatewayError,
    BackpressureError,
    BudgetExceededError,
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
    TokenBucketRateLimiter,
    configured_max_output_tokens,
    configured_routes_from_environment,
    default_gateway_from_environment,
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

    assert assembler.request.max_tokens == 70_000 + 40_960
    assert assembler.request.reserved_output_tokens == 40_960


def test_gateway_chat_model_records_content_derived_prompt_version() -> None:
    provider = FakeProviderAdapter({"deepseek-v4-flash": [_success("deepseek-v4-flash")]})
    gateway = ModelGateway(provider=provider)

    completion = GatewayChatModel(gateway).complete("system-v1", "hello")

    assert completion.model == "deepseek-v4-flash"
    assert gateway.cost_records[0].prompt_version.startswith("prompt-sha256-")
    assert gateway.cost_records[0].prompt_version != "runtime-prompt"


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
