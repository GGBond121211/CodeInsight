"""Model Gateway 的稳定错误分类。"""

from __future__ import annotations


class GatewayError(RuntimeError):
    code = "GATEWAY_ERROR"
    retryable = False
    fallback_allowed = False


class RateLimitGatewayError(GatewayError):
    code = "RATE_LIMIT"
    retryable = True
    fallback_allowed = True


class UpstreamGatewayError(GatewayError):
    code = "UPSTREAM_5XX"
    retryable = True
    fallback_allowed = True


class TimeoutGatewayError(GatewayError):
    code = "TIMEOUT"
    retryable = True
    fallback_allowed = True


class NetworkGatewayError(GatewayError):
    code = "NETWORK"
    retryable = True
    fallback_allowed = True


class InvalidResponseGatewayError(GatewayError):
    code = "INVALID_RESPONSE"
    retryable = True
    fallback_allowed = True


class AuthenticationGatewayError(GatewayError):
    code = "AUTHENTICATION"


class PermissionGatewayError(GatewayError):
    code = "PERMISSION"


class BadRequestGatewayError(GatewayError):
    code = "BAD_REQUEST"


class BudgetExceededError(GatewayError):
    code = "BUDGET_EXCEEDED"


class BackpressureError(GatewayError):
    code = "BACKPRESSURE"
    retryable = True


class NoCapableModelError(GatewayError):
    code = "NO_CAPABLE_MODEL"


class ContextOverflowGatewayError(GatewayError):
    """请求整体（输入 + 输出预留）超出工作上下文窗口。

    在调用前拒绝，而不是等上游返回 context_length_exceeded：上游报错既白花
    一次计费，又分不清是输入超限还是输出被截断。不重试，因为同一个超限请求
    重发结果不变；允许降级，因为换窗口更大的模型是正当出路。
    """

    code = "CONTEXT_OVERFLOW"
    retryable = False
    fallback_allowed = True


class CircuitOpenError(GatewayError):
    code = "CIRCUIT_OPEN"
    fallback_allowed = True
