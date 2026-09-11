import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from codeinsight.infrastructure.model_profiles import DEFAULT_MODEL_ID


def _free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _start(module: str, port: int, environment: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module}:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _wait_health(port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"service on {port} did not become healthy")


def _chat(port: int, request_id: str) -> dict[str, object]:
    payload = json.dumps(
        {
            "request_id": request_id,
            "scene": "explain",
            "messages": [{"role": "user", "content": "health probe"}],
            "estimated_input_tokens": 8,
            "reserved_output_tokens": 8,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_proxy_keeps_serving_after_one_gateway_process_stops() -> None:
    gateway_a_port, gateway_b_port, proxy_port = _free_port(), _free_port(), _free_port()
    environment = os.environ.copy()
    environment["CODEINSIGHT_GATEWAY_FAKE_MODE"] = "1"
    gateway_a = _start("codeinsight.infrastructure.gateway_server", gateway_a_port, environment)
    gateway_b = _start("codeinsight.infrastructure.gateway_server", gateway_b_port, environment)
    proxy_environment = environment | {
        "CODEINSIGHT_GATEWAY_UPSTREAMS": (
            f"http://127.0.0.1:{gateway_a_port},http://127.0.0.1:{gateway_b_port}"
        )
    }
    proxy = _start("codeinsight.infrastructure.gateway_proxy", proxy_port, proxy_environment)
    try:
        for port in (gateway_a_port, gateway_b_port, proxy_port):
            _wait_health(port)
        assert _chat(proxy_port, "before-stop")["model"] == DEFAULT_MODEL_ID

        gateway_a.terminate()
        gateway_a.wait(timeout=5)

        assert _chat(proxy_port, "after-stop")["model"] == DEFAULT_MODEL_ID
    finally:
        for process in (gateway_a, gateway_b, proxy):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
