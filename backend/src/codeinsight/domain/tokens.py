"""确定性的 token 估算，不引入 tokenizer 依赖。

放在 domain 层是因为 ingestion（切块时的 token 上限）和 application
（上下文预算）都要用它，而 ingestion 不能反向依赖 application。
单一实现可以避免两处口径漂移。
"""

from __future__ import annotations

BASIS = "utf8_bytes_div_4"


def estimate_tokens(text: str) -> int:
    """用确定性的保守近似估算 token。

    非空文本至少返回 1，避免极短行被估算成 0 token 而绕过预算检查。
    """
    if not text:
        return 0
    encoded_length = len(text.encode("utf-8"))
    return max(1, (encoded_length + 3) // 4)
