"""Step 7 本地反向代理：上游断开时尝试下一个无状态 Gateway 副本。"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from codeinsight.infrastructure.otel import get_telemetry


def _upstreams() -> tuple[str, ...]:
    raw = os.environ.get(
        "CODEINSIGHT_GATEWAY_UPSTREAMS",
        "http://127.0.0.1:8011,http://127.0.0.1:8012",
    )
    values = tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    if not values:
        raise RuntimeError("CODEINSIGHT_GATEWAY_UPSTREAMS 不能为空")
    return values


app = FastAPI(title="CodeInsight Gateway Proxy", version="2.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "upstreams": len(_upstreams())}


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request) -> JSONResponse:
    body = await request.body()
    failures: list[str] = []
    with get_telemetry().span("gateway_proxy", "forward"):
        for upstream in _upstreams():
            outgoing = urllib.request.Request(
                f"{upstream}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(outgoing, timeout=3) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    return JSONResponse(payload, status_code=response.status)
            except urllib.error.HTTPError as error:
                try:
                    payload = json.loads(error.read().decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {"detail": "gateway upstream returned HTTP error"}
                if error.code < 500:
                    return JSONResponse(payload, status_code=error.code)
                failures.append(f"HTTP_{error.code}")
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as error:
                failures.append(type(error).__name__)
    raise HTTPException(
        503,
        detail={"code": "ALL_GATEWAYS_UNAVAILABLE", "attempts": len(failures)},
    )


@app.get("/v1/usage/summary")
async def proxy_usage_summary(request: Request) -> JSONResponse:
    return _proxy_get_json(request, "/v1/usage/summary")


@app.get("/v1/usage/calls")
async def proxy_usage_calls(request: Request) -> JSONResponse:
    return _proxy_get_json(request, "/v1/usage/calls")


def _proxy_get_json(request: Request, path: str) -> JSONResponse:
    failures: list[str] = []
    query = request.url.query
    with get_telemetry().span("gateway_proxy", "forward_usage"):
        for upstream in _upstreams():
            target = f"{upstream}{path}"
            if query:
                target = f"{target}?{query}"
            outgoing = urllib.request.Request(target, method="GET")
            try:
                with urllib.request.urlopen(outgoing, timeout=3) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    return JSONResponse(payload, status_code=response.status)
            except urllib.error.HTTPError as error:
                try:
                    payload = json.loads(error.read().decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {"detail": "gateway upstream returned HTTP error"}
                if error.code < 500:
                    return JSONResponse(payload, status_code=error.code)
                failures.append(f"HTTP_{error.code}")
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as error:
                failures.append(type(error).__name__)
    raise HTTPException(
        503,
        detail={"code": "ALL_GATEWAYS_UNAVAILABLE", "attempts": len(failures)},
    )
