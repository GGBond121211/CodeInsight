"""对公开失败摘要和事件字段做确定性脱敏。"""

from __future__ import annotations

import re

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_BEARER = re.compile(r"(?i)\b(authorization\s*[:=]\s*bearer\s+)[^\s,;]+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)\b(api[_-]?key|secret|token|password|passwd|database_url)\b\s*[:=]\s*([^\s,;]+)"
)
_ENV_ASSIGNMENT = re.compile(r"(?im)^([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD))\s*=.*$")
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\(?:[^\s:\"'<>|]+\\)*[^\s:\"'<>|]*")
_UNIX_PATH = re.compile(r"(?<![\w.])/(?:home|Users|workspace|tmp|var/tmp)/[^\s:\"'<>|]+")


def redact_sensitive(value: str) -> str:
    """移除凭据、私钥和绝对路径，保留足够排障的错误类别。"""
    text = value.replace("\x00", "")
    text = _PRIVATE_KEY.sub("[REDACTED_PRIVATE_KEY]", text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _ENV_ASSIGNMENT.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _WINDOWS_PATH.sub("[REDACTED_PATH]", text)
    text = _UNIX_PATH.sub("[REDACTED_PATH]", text)
    return text
