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


class CircuitOpenError(GatewayError):
    code = "CIRCUIT_OPEN"
    fallback_allowed = True
