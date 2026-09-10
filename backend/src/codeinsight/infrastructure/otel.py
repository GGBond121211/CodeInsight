"""OpenTelemetry API 埋点与低基数 Prometheus 指标。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock

from opentelemetry import trace
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from codeinsight.domain.trace import assert_metric_labels, assert_recordable


@dataclass(frozen=True)
class SpanRecord:
    component: str
    operation: str
    attributes: dict[str, str]
    elapsed_milliseconds: float
    outcome: str


class Telemetry:
    """OTel 负责链路语义，内存记录和 Prometheus 让测试及本地演示可见。"""

    def __init__(self) -> None:
        self._tracer = trace.get_tracer("codeinsight", "2.1.0")
        self._records: list[SpanRecord] = []
        self._lock = RLock()
        self.registry = CollectorRegistry()
        labels = ["component", "operation", "outcome", "error_class"]
        assert_metric_labels(labels)
        self.calls = Counter(
            "codeinsight_operations_total",
            "CodeInsight component operations",
            labels,
            registry=self.registry,
        )
        self.latency = Histogram(
            "codeinsight_operation_duration_seconds",
            "CodeInsight component operation duration",
            labels,
            registry=self.registry,
        )
        self.gateway_cache_read_tokens = Counter(
            "codeinsight_gateway_cache_read_tokens_total",
            "Provider prompt cache read tokens",
            registry=self.registry,
        )
        self.gateway_cache_miss_tokens = Counter(
            "codeinsight_gateway_cache_miss_tokens_total",
            "Provider prompt cache miss tokens",
            registry=self.registry,
        )
        self.gateway_semantic_cache_hits = Counter(
            "codeinsight_gateway_semantic_cache_hits_total",
            "Exact semantic response cache hits",
            registry=self.registry,
        )

    @contextmanager
    def span(
        self,
        component: str,
        operation: str,
        *,
        attributes: dict[str, str] | None = None,
    ) -> Iterator[None]:
        values = dict(attributes or {})
        assert_recordable(values)
        started = time.perf_counter()
        outcome = "success"
        error_class = "none"
        with self._tracer.start_as_current_span(f"{component}.{operation}") as current:
            for key, value in values.items():
                current.set_attribute(f"codeinsight.{key}", value)
            try:
                yield
            except Exception as error:
                outcome = "error"
                error_class = type(error).__name__
                current.record_exception(error)
                raise
            finally:
                elapsed = time.perf_counter() - started
                metric_values = {
                    "component": component,
                    "operation": operation,
                    "outcome": outcome,
                    "error_class": error_class,
                }
                self.calls.labels(**metric_values).inc()
                self.latency.labels(**metric_values).observe(elapsed)
                with self._lock:
                    self._records.append(
                        SpanRecord(
                            component,
                            operation,
                            values,
                            elapsed * 1000,
                            outcome,
                        )
                    )

    def records(self) -> tuple[SpanRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def record_gateway_usage(
        self,
        *,
        cache_read_tokens: int = 0,
        cache_miss_tokens: int = 0,
        semantic_cache_hit: bool = False,
    ) -> None:
        """记录低基数 Gateway cache 指标，不把 request/model 放进 metric label。"""
        if cache_read_tokens < 0 or cache_miss_tokens < 0:
            raise ValueError("Gateway cache token 数不能为负")
        self.gateway_cache_read_tokens.inc(cache_read_tokens)
        self.gateway_cache_miss_tokens.inc(cache_miss_tokens)
        if semantic_cache_hit:
            self.gateway_semantic_cache_hits.inc()

    def metrics(self) -> bytes:
        return generate_latest(self.registry)


_TELEMETRY = Telemetry()


def get_telemetry() -> Telemetry:
    return _TELEMETRY
