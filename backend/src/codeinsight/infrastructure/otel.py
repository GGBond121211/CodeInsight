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
        # Worker 运行指标。标签只放低基数维度：task_kind / status。run_id、turn_id、
        # task_id、attempt_id 这类标识只进事件与 Span attribute——它们做 label 会让
        # 时间序列随请求量线性增长，Prometheus 最先扛不住的就是这个。
        run_labels = ["task_kind", "status"]
        task_labels = ["task_kind"]
        assert_metric_labels(run_labels)
        assert_metric_labels(task_labels)
        self.agent_run_attempts = Counter(
            "codeinsight_agent_run_attempts_total",
            "Agent Run Worker 的尝试次数",
            run_labels,
            registry=self.registry,
        )
        self.agent_run_duration = Histogram(
            "codeinsight_agent_run_duration_seconds",
            "Agent Run 一次尝试的执行耗时",
            run_labels,
            registry=self.registry,
        )
        self.agent_run_queue_wait = Histogram(
            "codeinsight_agent_run_queue_wait_seconds",
            "从写进 QUEUED 到被 Worker 领取的等待时间",
            task_labels,
            registry=self.registry,
        )
        self.agent_run_validation_wait = Histogram(
            "codeinsight_agent_run_validation_wait_seconds",
            "从登记固定校验到校验结束的等待时间",
            run_labels,
            registry=self.registry,
        )
        self.agent_run_model_calls = Counter(
            "codeinsight_agent_run_model_calls_total",
            "一次 Run 内的模型调用次数",
            task_labels,
            registry=self.registry,
        )
        self.agent_run_tool_calls = Counter(
            "codeinsight_agent_run_tool_calls_total",
            "一次 Run 内的工具调用次数",
            task_labels,
            registry=self.registry,
        )
        self.worker_recoveries = Counter(
            "codeinsight_worker_recoveries_total",
            "租约到期后被另一条消息接管的次数",
            task_labels,
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

    def record_agent_run(
        self,
        *,
        task_kind: str,
        status: str,
        execution_ms: int,
        model_calls: int = 0,
        tool_calls: int = 0,
        validation_wait_ms: int | None = None,
    ) -> None:
        """记一次 Worker 尝试的耗时与规模。

        计数由调用方从事件里数出来：模型调用、工具调用、校验等待原本就是事实，
        指标只是它们的投影，不另建一套账。
        """

        if execution_ms < 0:
            raise ValueError("execution_ms 不能为负")
        labels = {"task_kind": task_kind, "status": status}
        self.agent_run_attempts.labels(**labels).inc()
        self.agent_run_duration.labels(**labels).observe(execution_ms / 1000)
        if model_calls:
            self.agent_run_model_calls.labels(task_kind=task_kind).inc(model_calls)
        if tool_calls:
            self.agent_run_tool_calls.labels(task_kind=task_kind).inc(tool_calls)
        if validation_wait_ms is not None:
            self.agent_run_validation_wait.labels(**labels).observe(
                max(validation_wait_ms, 0) / 1000
            )

    def record_queue_wait(self, *, task_kind: str, wait_ms: int) -> None:
        """排队等待：从写进 QUEUED 到被领取。负值不记，也不猜。"""

        if wait_ms < 0:
            return
        self.agent_run_queue_wait.labels(task_kind=task_kind).observe(wait_ms / 1000)

    def record_worker_recovery(self, *, task_kind: str) -> None:
        """租约到期后的接管。它和「一次新的续跑」是两回事，必须分开记。"""

        self.worker_recoveries.labels(task_kind=task_kind).inc()

    def metrics(self) -> bytes:
        return generate_latest(self.registry)


_TELEMETRY = Telemetry()


def get_telemetry() -> Telemetry:
    return _TELEMETRY
