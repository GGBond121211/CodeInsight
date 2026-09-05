"""Run bounded Gateway concurrency checks with a local Fake Provider."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codeinsight.infrastructure.gateway_errors import BackpressureError
from codeinsight.infrastructure.model_gateway import (
    AdmissionController,
    GatewayRequest,
    InMemoryBudgetLedger,
    ModelGateway,
    TokenBucketRateLimiter,
)
from codeinsight.infrastructure.provider_adapters import ProviderRequest, ProviderResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "STEP8-MINIMUM-RELEASE"


class SlowFakeProvider:
    def __init__(self, delay_seconds: float = 0.02) -> None:
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_seconds)
            return ProviderResponse(
                content='{"status":"fake-concurrency-ok"}',
                tool_calls=(),
                model=request.model,
                input_tokens=8,
                output_tokens=4,
                cached_input_tokens=0,
                finish_reason="stop",
            )
        finally:
            with self._lock:
                self.active -= 1


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(percentile * len(ordered) + 0.9999) - 1))
    return round(ordered[index], 3)


def _request(index: int) -> GatewayRequest:
    return GatewayRequest(
        request_id=f"step8-fake-concurrency-{index}",
        tenant_id="default",
        user_id="local",
        scene="explain",
        prompt_version="step8-fake-concurrency-v1",
        messages=({"role": "user", "content": f"probe-{index}"},),
        estimated_input_tokens=8,
        reserved_output_tokens=8,
    )


def _run_burst(burst: int) -> dict[str, object]:
    provider = SlowFakeProvider()
    gateway = ModelGateway(
        provider=provider,
        admission=AdmissionController(max_concurrency=8),
        budget_ledger=InMemoryBudgetLedger(default_limit=1_000_000),
        rate_limiter=TokenBucketRateLimiter(
            capacity=1_000_000,
            refill_tokens_per_second=1_000_000,
        ),
        max_retries=0,
    )

    def invoke(index: int) -> dict[str, object]:
        started = time.perf_counter()
        try:
            response = gateway.complete(_request(index))
        except BackpressureError:
            return {"status": "backpressure", "latency_ms": (time.perf_counter() - started) * 1000}
        return {
            "status": "accepted",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "model": response.model,
        }

    with ThreadPoolExecutor(max_workers=burst) as pool:
        results = list(pool.map(invoke, range(burst)))
    accepted = [item for item in results if item["status"] == "accepted"]
    latencies = [float(item["latency_ms"]) for item in accepted]
    return {
        "burst": burst,
        "admission_limit": 8,
        "accepted": len(accepted),
        "backpressure": burst - len(accepted),
        "provider_max_active": provider.max_active,
        "accepted_p50_ms": _percentile(latencies, 0.50),
        "accepted_p95_ms": _percentile(latencies, 0.95),
        "provider": "local_slow_fake_only",
    }


def run() -> dict[str, object]:
    run_dir = sorted(OUTPUT_ROOT.glob("*"))[-1]
    rows = [_run_burst(burst) for burst in (50, 100, 200)]
    output = {
        "schema_version": 1,
        "experiment_id": "STEP8-FAKE-CONCURRENCY",
        "run_id": run_dir.name,
        "rows": rows,
        "interpretation": [
            (
                "仅验证本地 AdmissionController/backpressure 契约，"
                "不代表任何真实 Provider 吞吐或 P95。"
            ),
            "本轮不运行 300/450 burst，也不调用真实模型。",
        ],
    }
    path = run_dir / "fake_concurrency.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
    return output


if __name__ == "__main__":
    run()
