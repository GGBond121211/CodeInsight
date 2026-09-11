from fastapi.testclient import TestClient

from codeinsight.infrastructure.gateway_server import create_gateway_app
from codeinsight.infrastructure.model_gateway import ModelGateway
from codeinsight.infrastructure.model_profiles import (
    DEFAULT_MODEL_ID,
    default_model_registry,
)
from codeinsight.infrastructure.provider_adapters import FakeProviderAdapter, ProviderResponse


def _gateway() -> ModelGateway:
    return ModelGateway(
        provider=FakeProviderAdapter(
            {
                DEFAULT_MODEL_ID: [
                    ProviderResponse("ok", (), DEFAULT_MODEL_ID, 5, 2, 0, "stop")
                ]
            }
        )
    )


def test_two_stateless_gateway_instances_can_serve_requests() -> None:
    clients = [
        TestClient(create_gateway_app(_gateway())),
        TestClient(create_gateway_app(_gateway())),
    ]
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "scene": "explain",
        "request_id": "req-service",
        "estimated_input_tokens": 5,
        "reserved_output_tokens": 5,
    }

    responses = [client.post("/v1/chat/completions", json=payload) for client in clients]

    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["model"] for response in responses] == [
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_ID,
    ]
    assert clients[0].get("/metrics").status_code == 200
    models = clients[0].get("/v1/models").json()["data"]
    assert len(models) == len(default_model_registry().all())
    assert next(item for item in models if item["quality_tier"] == "best")["id"] == DEFAULT_MODEL_ID
