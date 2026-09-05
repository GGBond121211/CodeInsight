"""场景路由、降级、预算、熔断、缓存和成本对账的本地 Model Gateway。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache

from openai import OpenAI

from codeinsight.agent.tool_loop import ToolModelResponse
from codeinsight.application.context_budget import estimate_tokens
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.trace import MODEL_CALLED, MODEL_RESULT, RunEvent
from codeinsight.infrastructure.event_log import InMemoryEventLog
from codeinsight.infrastructure.gateway_errors import (
    BackpressureError,
    BudgetExceededError,
    CircuitOpenError,
    GatewayError,
    InvalidResponseGatewayError,
    NoCapableModelError,
)
from codeinsight.infrastructure.model_profiles import (
    DEFAULT_MODEL_ID,
    ModelProfile,
    ModelRegistry,
    RouteProfile,
    default_model_registry,
    default_routes,
)
from codeinsight.infrastructure.otel import Telemetry, get_telemetry
from codeinsight.infrastructure.provider_adapters import (
    OpenAIProviderAdapter,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)


@dataclass(frozen=True)
class CacheContext:
    repo_id: str
    repo_fingerprint: str
    index_version: str
    contains_workspace_state: bool
    personalized: bool


@dataclass(frozen=True)
class GatewayRequest:
    request_id: str
    tenant_id: str
    user_id: str
    scene: str
    prompt_version: str
    messages: tuple[Mapping[str, object], ...]
    estimated_input_tokens: int
    reserved_output_tokens: int
    required_capabilities: frozenset[str] = frozenset({"text"})
    tools: tuple[Mapping[str, object], ...] = ()
    response_format: Mapping[str, object] | None = None
    cache_context: CacheContext | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.tenant_id or not self.user_id:
            raise ValueError("request_id/tenant_id/user_id 不能为空")
        if self.estimated_input_tokens < 0 or self.reserved_output_tokens < 0:
            raise ValueError("Token 估算不能为负")


@dataclass(frozen=True)
class AttemptTrace:
    request_id: str
    attempt_id: str
    tenant_id: str
    user_id: str
    scene: str
    prompt_version: str
    provider: str
    model_tier: str
    model: str
    price_version: str
    latency_milliseconds: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    fallback_reason: str | None
    error_class: str | None


@dataclass(frozen=True)
class CostRecord:
    request_id: str
    attempt_id: str
    tenant_id: str
    user_id: str
    scene: str
    prompt_version: str
    provider: str
    model_tier: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    price_version: str
    total_stars: Decimal
    latency_milliseconds: float
    ttft_milliseconds: float | None
    fallback_reason: str | None
    error_class: str | None = None


@dataclass(frozen=True)
class GatewayResponse:
    content: str | None
    tool_calls: tuple
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    attempts: tuple[AttemptTrace, ...]
    cost: CostRecord
    cache_hit: bool = False


@dataclass(frozen=True)
class _AttemptFailure(Exception):
    error: GatewayError
    trace: AttemptTrace
    cost: CostRecord


@dataclass(frozen=True)
class _Reservation:
    reservation_id: str
    tenant_id: str
    reserved_tokens: int


class InMemoryBudgetLedger:
    """estimate -> reserve -> actual -> reconcile 的可执行定义。"""

    def __init__(self, *, default_limit: int = 200_000) -> None:
        self.default_limit = default_limit
        self._used: dict[str, int] = {}
        self._reserved: dict[str, _Reservation] = {}
        self._lock = threading.RLock()
        self.reconciliations: list[tuple[str, int, int]] = []
        self.underestimation_warnings: list[tuple[str, int, int]] = []

    def reserve(self, tenant_id: str, tokens: int) -> _Reservation:
        with self._lock:
            outstanding = sum(
                item.reserved_tokens
                for item in self._reserved.values()
                if item.tenant_id == tenant_id
            )
            used = self._used.get(tenant_id, 0)
            if used + outstanding + tokens > self.default_limit:
                raise BudgetExceededError("预算预留失败；请求未进入模型调用")
            reservation = _Reservation(uuid.uuid4().hex, tenant_id, tokens)
            self._reserved[reservation.reservation_id] = reservation
            return reservation

    def reconcile(self, reservation: _Reservation, actual_tokens: int) -> None:
        with self._lock:
            present = self._reserved.pop(reservation.reservation_id, None)
            if present is None:
                raise ValueError("预算预留不存在或已对账")
            self._used[reservation.tenant_id] = (
                self._used.get(reservation.tenant_id, 0) + actual_tokens
            )
            self.reconciliations.append(
                (reservation.reservation_id, reservation.reserved_tokens, actual_tokens)
            )
            if actual_tokens > reservation.reserved_tokens * 1.2 + 32:
                self.underestimation_warnings.append(
                    (reservation.reservation_id, reservation.reserved_tokens, actual_tokens)
                )


class TokenBucketRateLimiter:
    """按 tenant + scene 限制短时间 Token 流量，避免无限冲击上游。"""

    def __init__(
        self,
        *,
        capacity: int = 200_000,
        refill_tokens_per_second: float = 10_000,
        clock=time.monotonic,
    ) -> None:
        if capacity <= 0 or refill_tokens_per_second <= 0:
            raise ValueError("令牌桶容量和补充速率必须为正")
        self.capacity = capacity
        self.refill_rate = refill_tokens_per_second
        self.clock = clock
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.RLock()

    def consume(self, tenant_id: str, scene: str, tokens: int) -> None:
        key = (tenant_id, scene)
        now = self.clock()
        with self._lock:
            available, updated_at = self._buckets.get(key, (float(self.capacity), now))
            available = min(self.capacity, available + (now - updated_at) * self.refill_rate)
            if tokens > available:
                self._buckets[key] = (available, now)
                raise BackpressureError("route/tenant Token Bucket 已耗尽")
            self._buckets[key] = (available - tokens, now)


class CircuitBreaker:
    def __init__(self, *, failure_threshold: int = 2, cooldown_seconds: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def allow(self, model: str) -> bool:
        opened = self._opened_at.get(model)
        if opened is None:
            return True
        if time.monotonic() - opened >= self.cooldown_seconds:
            self._failures[model] = 0
            self._opened_at.pop(model, None)
            return True
        return False

    def success(self, model: str) -> None:
        self._failures[model] = 0
        self._opened_at.pop(model, None)

    def failure(self, model: str) -> None:
        count = self._failures.get(model, 0) + 1
        self._failures[model] = count
        if count >= self.failure_threshold:
            self._opened_at[model] = time.monotonic()


class AdmissionController:
    def __init__(self, *, max_concurrency: int = 8) -> None:
        self.max_concurrency = max_concurrency
        self._active = 0
        self._lock = threading.RLock()

    def acquire(self) -> None:
        with self._lock:
            if self._active >= self.max_concurrency:
                raise BackpressureError("Gateway 并发已满")
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active -= 1


class GatewaySemanticCache:
    KEY_VERSION = "gateway-semantic-v1"

    def __init__(self) -> None:
        self._values: dict[str, GatewayResponse] = {}

    def key_for(self, request: GatewayRequest, model_id: str) -> str | None:
        context = request.cache_context
        if context is None or context.contains_workspace_state:
            return None
        parts = [
            request.tenant_id,
            context.repo_id,
            context.repo_fingerprint,
            context.index_version,
            request.prompt_version,
            model_id,
            request.scene,
            self.KEY_VERSION,
        ]
        if context.personalized:
            parts.append(request.user_id)
        serialized = json.dumps(request.messages, ensure_ascii=False, sort_keys=True, default=str)
        parts.append(hashlib.sha256(serialized.encode("utf-8")).hexdigest())
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str | None) -> GatewayResponse | None:
        return None if key is None else self._values.get(key)

    def set(self, key: str | None, response: GatewayResponse) -> None:
        if key is not None:
            self._values[key] = response


class ModelGateway:
    def __init__(
        self,
        *,
        provider: ProviderAdapter,
        registry: ModelRegistry | None = None,
        routes: Mapping[str, RouteProfile] | None = None,
        budget_ledger: InMemoryBudgetLedger | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        admission: AdmissionController | None = None,
        semantic_cache: GatewaySemanticCache | None = None,
        telemetry: Telemetry | None = None,
        event_log=None,
        cost_store=None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        max_retries: int = 1,
    ) -> None:
        self.provider = provider
        self.registry = registry or default_model_registry()
        self.routes = dict(routes or default_routes(self.registry))
        self.budget = budget_ledger or InMemoryBudgetLedger()
        self.circuit = circuit_breaker or CircuitBreaker()
        self.admission = admission or AdmissionController()
        self.semantic_cache = semantic_cache or GatewaySemanticCache()
        self.telemetry = telemetry or get_telemetry()
        self.event_log = event_log or InMemoryEventLog()
        self.cost_store = cost_store
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self.max_retries = max_retries
        self.cost_records: list[CostRecord] = []

    def complete(self, request: GatewayRequest) -> GatewayResponse:
        route = self.routes.get(request.scene)
        if route is None:
            raise NoCapableModelError(f"未登记场景：{request.scene}")
        required = request.required_capabilities or route.required_capabilities
        candidates = (route.primary_model_id, *route.fallback_model_ids)
        candidates = tuple(
            model_id
            for model_id in candidates
            if required <= self.registry.require(model_id).capabilities
        )
        if not candidates:
            raise NoCapableModelError("没有满足当前能力要求的模型")

        primary_cache_key = self.semantic_cache.key_for(request, candidates[0])
        cached = self.semantic_cache.get(primary_cache_key)
        if cached is not None:
            return replace(cached, cache_hit=True)

        self.admission.acquire()
        attempts: list[AttemptTrace] = []
        fallback_reason: str | None = None
        last_error: GatewayError | None = None
        try:
            for model_id in candidates:
                profile = self.registry.require(model_id)
                if not self.circuit.allow(model_id):
                    last_error = CircuitOpenError(f"{model_id} 熔断中")
                    fallback_reason = last_error.code
                    continue
                ordinary_retries = self.max_retries
                format_repairs = 1
                while True:
                    response_trace: AttemptTrace | None = None
                    response_cost: CostRecord | None = None
                    try:
                        response, response_trace, response_cost = self._attempt(
                            request, profile, fallback_reason=fallback_reason
                        )
                        self._validate_response(request, response)
                    except InvalidResponseGatewayError as error:
                        if response_trace is not None and response_cost is not None:
                            attempts.append(replace(response_trace, error_class=error.code))
                            failed_cost = replace(response_cost, error_class=error.code)
                            self._record_cost(failed_cost)
                            self._emit(
                                request,
                                MODEL_RESULT,
                                {
                                    "attempt_id": response_trace.attempt_id,
                                    "scene": request.scene,
                                    "model": profile.model_id,
                                    "outcome": "error",
                                    "error_class": error.code,
                                },
                            )
                        else:
                            attempts.append(
                                self._error_trace(request, profile, error, fallback_reason)
                            )
                        last_error = error
                        if format_repairs > 0:
                            format_repairs -= 1
                            continue
                        self.circuit.failure(model_id)
                        fallback_reason = error.code
                        break
                    except _AttemptFailure as failure:
                        error = failure.error
                        attempts.append(failure.trace)
                        self._record_cost(failure.cost)
                        last_error = error
                        if not error.fallback_allowed:
                            raise error
                        self.circuit.failure(model_id)
                        if error.retryable and ordinary_retries > 0:
                            ordinary_retries -= 1
                            continue
                        fallback_reason = error.code
                        break
                    self.circuit.success(model_id)
                    assert response_trace is not None
                    assert response_cost is not None
                    self._emit(
                        request,
                        MODEL_RESULT,
                        {
                            "attempt_id": response_trace.attempt_id,
                            "scene": request.scene,
                            "model": response.model,
                            "outcome": "success",
                            "input_tokens": str(response.input_tokens),
                            "output_tokens": str(response.output_tokens),
                        },
                    )
                    attempts.append(response_trace)
                    self._record_cost(response_cost)
                    result = GatewayResponse(
                        response.content,
                        response.tool_calls,
                        response.model,
                        response.input_tokens,
                        response.output_tokens,
                        response.cached_input_tokens,
                        tuple(attempts),
                        response_cost,
                    )
                    if response.model == candidates[0]:
                        self.semantic_cache.set(primary_cache_key, result)
                    return result
            if last_error is not None:
                raise last_error
            raise NoCapableModelError("路由没有可调用模型")
        finally:
            self.admission.release()

    def _attempt(
        self, request: GatewayRequest, profile: ModelProfile, *, fallback_reason: str | None
    ) -> tuple[ProviderResponse, AttemptTrace, CostRecord]:
        self.rate_limiter.consume(
            request.tenant_id,
            request.scene,
            request.estimated_input_tokens + request.reserved_output_tokens,
        )
        reservation = self.budget.reserve(
            request.tenant_id,
            request.estimated_input_tokens + request.reserved_output_tokens,
        )
        attempt_id = uuid.uuid4().hex
        started = time.perf_counter()
        self._emit(
            request,
            MODEL_CALLED,
            {
                "attempt_id": attempt_id,
                "scene": request.scene,
                "model": profile.model_id,
                "fallback_reason": fallback_reason or "none",
            },
        )
        try:
            with self.telemetry.span(
                "gateway",
                "model_attempt",
                attributes={
                    "request_id": request.request_id,
                    "attempt_id": attempt_id,
                    "tenant_id": request.tenant_id,
                    "user_id": request.user_id,
                    "scene": request.scene,
                    "model": profile.model_id,
                },
            ):
                response = self.provider.invoke(
                    ProviderRequest(
                        profile.model_id,
                        request.messages,
                        request.tools,
                        request.response_format,
                        request.reserved_output_tokens or None,
                    )
                )
        except GatewayError as error:
            elapsed = (time.perf_counter() - started) * 1000
            self.budget.reconcile(reservation, 0)
            self._emit(
                request,
                MODEL_RESULT,
                {
                    "attempt_id": attempt_id,
                    "scene": request.scene,
                    "model": profile.model_id,
                    "outcome": "error",
                    "error_class": error.code,
                },
            )
            trace = AttemptTrace(
                request.request_id,
                attempt_id,
                request.tenant_id,
                request.user_id,
                request.scene,
                request.prompt_version,
                profile.provider,
                profile.quality_tier,
                profile.model_id,
                profile.price_version,
                elapsed,
                0,
                0,
                0,
                fallback_reason,
                error.code,
            )
            cost = CostRecord(
                request_id=request.request_id,
                attempt_id=attempt_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                scene=request.scene,
                prompt_version=request.prompt_version,
                provider=profile.provider,
                model_tier=profile.quality_tier,
                model=profile.model_id,
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
                price_version=profile.price_version,
                total_stars=Decimal(0),
                latency_milliseconds=elapsed,
                ttft_milliseconds=None,
                fallback_reason=fallback_reason,
                error_class=error.code,
            )
            raise _AttemptFailure(error, trace, cost) from error
        elapsed = (time.perf_counter() - started) * 1000
        actual = response.input_tokens + response.output_tokens
        self.budget.reconcile(reservation, actual)
        trace = AttemptTrace(
            request.request_id,
            attempt_id,
            request.tenant_id,
            request.user_id,
            request.scene,
            request.prompt_version,
            profile.provider,
            profile.quality_tier,
            response.model,
            profile.price_version,
            elapsed,
            response.input_tokens,
            response.output_tokens,
            response.cached_input_tokens,
            fallback_reason,
            None,
        )
        cost = _cost_record(request, profile, response, attempt_id, elapsed, fallback_reason)
        return response, trace, cost

    def _emit(self, request: GatewayRequest, event_type: str, payload: dict[str, str]) -> None:
        run_id = request.run_id or request.request_id
        sequence = self.event_log.next_sequence(run_id)
        self.event_log.append(
            RunEvent(
                event_id=f"{run_id}:{sequence}",
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                occurred_at_epoch_ms=int(time.time() * 1000),
                payload=payload,
            )
        )

    def _record_cost(self, record: CostRecord) -> None:
        self.cost_records.append(record)
        if self.cost_store is not None:
            self.cost_store.record(record)

    @staticmethod
    def _validate_response(request: GatewayRequest, response: ProviderResponse) -> None:
        if request.tools and response.content is None and not response.tool_calls:
            raise InvalidResponseGatewayError("工具路线返回了空响应")
        if request.response_format is not None:
            try:
                parsed = json.loads(response.content or "")
            except json.JSONDecodeError:
                raise InvalidResponseGatewayError("结构化输出不是有效 JSON") from None
            if not isinstance(parsed, dict):
                raise InvalidResponseGatewayError("结构化输出必须是 JSON object")

    @staticmethod
    def _error_trace(
        request: GatewayRequest,
        profile: ModelProfile,
        error: GatewayError,
        fallback_reason: str | None,
    ) -> AttemptTrace:
        return AttemptTrace(
            request.request_id,
            uuid.uuid4().hex,
            request.tenant_id,
            request.user_id,
            request.scene,
            request.prompt_version,
            profile.provider,
            profile.quality_tier,
            profile.model_id,
            profile.price_version,
            0,
            0,
            0,
            0,
            fallback_reason,
            error.code,
        )


def _cost_record(
    request: GatewayRequest,
    profile: ModelProfile,
    response: ProviderResponse,
    attempt_id: str,
    elapsed: float,
    fallback_reason: str | None,
) -> CostRecord:
    uncached = max(0, response.input_tokens - response.cached_input_tokens)
    stars = (
        Decimal(uncached) * profile.input_price_per_million
        + Decimal(response.cached_input_tokens) * profile.cached_input_price_per_million
        + Decimal(response.output_tokens) * profile.output_price_per_million
    ) / Decimal(1_000_000)
    return CostRecord(
        request.request_id,
        attempt_id,
        request.tenant_id,
        request.user_id,
        request.scene,
        request.prompt_version,
        profile.provider,
        profile.quality_tier,
        response.model,
        response.input_tokens,
        response.output_tokens,
        response.cached_input_tokens,
        profile.price_version,
        stars,
        elapsed,
        None,
        fallback_reason,
    )


@lru_cache(maxsize=1)
def default_gateway_from_environment() -> ModelGateway:
    """进程内共享状态；熔断、配额、缓存不能每个 HTTP 请求重新创建。"""
    if os.environ.get("CODEINSIGHT_GATEWAY_FAKE_MODE") == "1":
        from codeinsight.infrastructure.provider_adapters import StaticFakeProviderAdapter

        return ModelGateway(provider=StaticFakeProviderAdapter())
    api_key = os.environ.get("CODEINSIGHT_API_KEY", "").strip()
    if not api_key:
        from codeinsight.domain.errors import ModelConfigurationError

        raise ModelConfigurationError("必须配置 CODEINSIGHT_API_KEY")
    base_url = os.environ.get("CODEINSIGHT_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    event_log = None
    cost_store = None
    from codeinsight.infrastructure.db.engine import (
        MySqlConfig,
        create_all_tables,
        create_db_engine,
        create_session_factory,
    )

    mysql = MySqlConfig.from_env()
    if mysql is not None:
        from codeinsight.infrastructure.db.stores import MySqlEventLog, MySqlGatewayCostStore

        engine = create_db_engine(mysql)
        create_all_tables(engine)
        session_factory = create_session_factory(engine)
        event_log = MySqlEventLog(session_factory)
        cost_store = MySqlGatewayCostStore(session_factory)
    return ModelGateway(
        provider=OpenAIProviderAdapter(client), event_log=event_log, cost_store=cost_store
    )


class GatewayChatModel:
    """保持现有应用 Model 接口不变，把真实调用统一送入 Gateway。"""

    model = DEFAULT_MODEL_ID

    def __init__(self, gateway: ModelGateway, *, context_assembler=None) -> None:
        self.gateway = gateway
        self._context_assembler = context_assembler

    @classmethod
    def from_environment(cls, *, context_assembler=None) -> GatewayChatModel:
        return cls(default_gateway_from_environment(), context_assembler=context_assembler)

    def complete(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
        request_user_prompt = user_prompt
        estimated = estimate_tokens(system_prompt + user_prompt)
        if self._context_assembler is not None:
            from codeinsight.application.context_assembler import ContextRequest

            assembly = self._context_assembler.assemble(
                ContextRequest(
                    system_safety=system_prompt,
                    user_goal="",
                    user_code_task=user_prompt,
                    model_version=self.model,
                    strategy_version="context-assembler-v1",
                )
            )
            request_user_prompt = assembly.user_text
            estimated = assembly.fitted.estimate.input_tokens
        response = self.gateway.complete(
            GatewayRequest(
                uuid.uuid4().hex,
                "default",
                "local",
                "explain",
                _prompt_version(system_prompt),
                (
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request_user_prompt},
                ),
                estimated,
                1000,
                frozenset({"text"}),
                response_format={"type": "json_object"},
            )
        )
        return ModelCompletion(
            response.content or "",
            response.model,
            response.input_tokens,
            response.output_tokens,
            estimated,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
        from codeinsight.infrastructure.openai_chat import parse_model_answer

        result = self.complete(system_prompt, user_prompt)
        return parse_model_answer(
            result.content,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def complete_with_tools(
        self,
        messages: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
        tools: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    ) -> ToolModelResponse:
        response = self.gateway.complete(
            GatewayRequest(
                uuid.uuid4().hex,
                "default",
                "local",
                "change-plan",
                "tool-loop-v1",
                tuple(messages),
                estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str)),
                1000,
                frozenset({"text", "tools", "structured_output"}),
                tuple(tools),
            )
        )
        return ToolModelResponse(
            response.content,
            response.tool_calls,
            response.model,
            response.input_tokens,
            response.output_tokens,
        )


def _prompt_version(system_prompt: str) -> str:
    """旧 Prompt 未显式传版本时用内容 Hash，避免成本记录写模糊常量。"""
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    return f"prompt-sha256-{digest}"
