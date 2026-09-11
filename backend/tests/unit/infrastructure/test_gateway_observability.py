from types import SimpleNamespace

from fastapi.testclient import TestClient

from codeinsight.infrastructure.gateway_server import create_gateway_app
from codeinsight.infrastructure.model_gateway import (
    CacheContext,
    GatewayRequest,
    ModelGateway,
    configured_routes_from_environment,
)
from codeinsight.infrastructure.model_profiles import DEFAULT_MODEL_ID
from codeinsight.infrastructure.otel import Telemetry
from codeinsight.infrastructure.provider_adapters import (
    OpenAIProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)


class _FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=None,
                        reasoning_content="provider debug reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=5,
                prompt_cache_hit_tokens=70,
                prompt_cache_miss_tokens=30,
            ),
        )


class _FakeClient:
    chat = SimpleNamespace(completions=_FakeCompletions())


def test_openai_adapter_reads_deepseek_native_cache_usage() -> None:
    response = OpenAIProviderAdapter(_FakeClient()).invoke(
        ProviderRequest(model="deepseek-v4-flash", messages=({"role": "user"},))
    )

    assert response.input_tokens == 100
    assert response.cached_input_tokens == 70
    assert response.cache_miss_tokens == 30
    assert response.effective_cache_miss_tokens == 30
    assert response.usage_source == "provider_native"
    assert response.cache_hit_ratio == 0.7
    assert response.reasoning_content == "provider debug reasoning"


def _request(**overrides: object) -> GatewayRequest:
    values: dict[str, object] = {
        "request_id": "req-observe",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "scene": "explain",
        "prompt_version": "prompt-v1",
        "messages": ({"role": "user", "content": "same"},),
        "estimated_input_tokens": 100,
        "reserved_output_tokens": 20,
        "required_capabilities": frozenset({"text"}),
        "cache_context": CacheContext("repo", "fingerprint", "index-v1", False, False),
    }
    values.update(overrides)
    return GatewayRequest(**values)  # type: ignore[arg-type]


def _cached_response() -> ProviderResponse:
    return ProviderResponse(
        "ok",
        (),
        "deepseek-v4-flash",
        100,
        5,
        70,
        "stop",
        cache_miss_tokens=30,
        usage_source="provider_native",
    )


def test_semantic_cache_key_includes_tools_and_response_format() -> None:
    gateway = ModelGateway(provider=lambda request: _cached_response())  # type: ignore[arg-type]
    base = _request()

    with_tools = _request(
        tools=({"type": "function", "function": {"name": "read_file"}},)
    )
    with_format = _request(response_format={"type": "json_object"})

    base_key = gateway.semantic_cache.key_for(base, "deepseek-v4-flash")
    tools_key = gateway.semantic_cache.key_for(with_tools, "deepseek-v4-flash")
    format_key = gateway.semantic_cache.key_for(with_format, "deepseek-v4-flash")
    assert base_key != tools_key
    assert base_key != format_key


def test_gateway_usage_endpoints_expose_cache_breakdown_without_prompt() -> None:
    class Provider:
        def invoke(self, request):
            return _cached_response()

    gateway = ModelGateway(provider=Provider(), telemetry=Telemetry())
    client = TestClient(create_gateway_app(gateway))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "same"}],
            "estimated_input_tokens": 100,
            "reserved_output_tokens": 20,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["usage"]["prompt_cache_hit_tokens"] == 70
    assert payload["usage"]["prompt_cache_miss_tokens"] == 30
    assert len(payload["gateway"]["request_fingerprint"]) == 64

    summary = client.get("/v1/usage/summary").json()
    assert summary["provider_attempts"] == 1
    assert summary["cache_read_tokens"] == 70
    assert summary["cache_miss_tokens"] == 30
    detail = client.get("/v1/usage/calls?limit=1").json()["data"][0]
    assert detail["cache_read_tokens"] == 70
    assert detail["cache_miss_tokens"] == 30
    assert "content" not in detail
    metrics = client.get("/metrics").text
    assert "codeinsight_gateway_cache_read_tokens_total 70.0" in metrics
    assert "codeinsight_gateway_cache_miss_tokens_total 30.0" in metrics


def test_gateway_health_reports_the_runtime_route_configuration(monkeypatch) -> None:
    monkeypatch.delenv("CODEINSIGHT_FALLBACK_MODELS", raising=False)
    gateway = ModelGateway(
        provider=OpenAIProviderAdapter(_FakeClient()),
        routes=configured_routes_from_environment(),
    )
    client = TestClient(create_gateway_app(gateway))

    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["routes"]["explain"] == {
        "primary": DEFAULT_MODEL_ID,
        "fallbacks": [],
    }
