from fastapi.testclient import TestClient

from codeinsight.infrastructure.gateway_server import create_gateway_app
from codeinsight.infrastructure.model_gateway import ModelGateway
from codeinsight.infrastructure.provider_adapters import FakeProviderAdapter, ProviderResponse


def _gateway() -> ModelGateway:
    return ModelGateway(
        provider=FakeProviderAdapter(
            {
                "deepseek-v4-flash": [
                    ProviderResponse("ok", (), "deepseek-v4-flash", 5, 2, 0, "stop")
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
        "deepseek-v4-flash",
        "deepseek-v4-flash",
    ]
    assert clients[0].get("/metrics").status_code == 200
    models = clients[0].get("/v1/models").json()["data"]
    assert len(models) == 12
    assert next(item for item in models if item["quality_tier"] == "best")["id"] == (
        "deepseek-v4-flash"
    )
