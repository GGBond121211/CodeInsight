"""阿里云百炼文本 Rerank 适配器。

Rerank 不是 Chat Completions，也不是 Embedding：它接收一个 query 和一组
documents，返回这些 documents 的排序 index 与 relevance_score。适配器只负责
供应商协议和响应校验，候选证据到源码行号的映射仍由 retrieval 层完成。
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from codeinsight.domain.errors import ModelCallError, ModelConfigurationError, ModelResponseError

DEFAULT_RERANK_MODEL = "qwen3.7-text-rerank"
MAX_DOCUMENTS = 500
RERANK_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class RerankResult:
    """模型返回的一条排序结果。index 对应请求 documents 的零基下标。"""

    index: int
    relevance_score: float


@dataclass(frozen=True)
class RerankCallTelemetry:
    """单次 Rerank 的可公开运行指标，不包含请求体或凭据。"""

    request_id: str
    model: str
    document_count: int
    top_n: int
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_input_tokens: int
    status: str
    error_type: str | None = None


class Reranker(Protocol):
    """Rerank 能力的最小依赖注入契约。"""

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> tuple[RerankResult, ...]: ...


class OpenAITextReranker:
    """调用阿里云百炼兼容 Rerank API 的文本重排模型。"""

    def __init__(self, *, api_key: str, endpoint: str, model: str) -> None:
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        self.last_call: RerankCallTelemetry | None = None

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> OpenAITextReranker:
        source = os.environ if environ is None else environ
        # 用户已确认 Rerank 与 Embedding 共用 Key；专用变量优先，便于以后拆分。
        api_key = source.get("CODEINSIGHT_RERANK_API_KEY") or source.get(
            "CODEINSIGHT_EMBEDDING_API_KEY"
        )
        base_url = source.get("CODEINSIGHT_RERANK_BASE_URL") or source.get(
            "CODEINSIGHT_EMBEDDING_BASE_URL"
        )
        model = source.get("CODEINSIGHT_RERANK_MODEL") or DEFAULT_RERANK_MODEL
        if not api_key:
            raise ModelConfigurationError(
                "必须配置 CODEINSIGHT_RERANK_API_KEY 或 CODEINSIGHT_EMBEDDING_API_KEY"
            )
        if not base_url:
            raise ModelConfigurationError(
                "必须配置 CODEINSIGHT_RERANK_BASE_URL 或 CODEINSIGHT_EMBEDDING_BASE_URL"
            )
        return cls(
            api_key=api_key,
            endpoint=_rerank_endpoint(base_url),
            model=model,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> tuple[RerankResult, ...]:
        if not query.strip():
            raise ValueError("Rerank query 不能为空")
        if not documents:
            return ()
        if len(documents) > MAX_DOCUMENTS:
            raise ValueError(f"Rerank 单次最多接受 {MAX_DOCUMENTS} 个文档")
        if top_n <= 0 or top_n > len(documents):
            raise ValueError("Rerank top_n 必须在 1 到文档数量之间")
        if any(not document.strip() for document in documents):
            raise ValueError("Rerank documents 不能包含空文本")

        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "return_documents": False,
        }
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        estimated_input_tokens = max(
            1,
            sum(len(document.encode("utf-8")) for document in documents)
            // 4
            + len(query.encode("utf-8")) // 4,
        )
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=RERANK_TIMEOUT_SECONDS) as response:
                raw = response.read()
                response_request_id = _header_value(response, "x-request-id")
        except HTTPError as error:
            self.last_call = _telemetry(
                request_id=request_id,
                response_request_id=None,
                model=self.model,
                document_count=len(documents),
                top_n=top_n,
                started=started,
                estimated_input_tokens=estimated_input_tokens,
                status=f"HTTP_{error.code}",
                error_type="HTTPError",
            )
            raise ModelCallError(f"Rerank 请求失败（HTTP {error.code}）") from error
        except (URLError, TimeoutError, OSError) as error:
            self.last_call = _telemetry(
                request_id=request_id,
                response_request_id=None,
                model=self.model,
                document_count=len(documents),
                top_n=top_n,
                started=started,
                estimated_input_tokens=estimated_input_tokens,
                status="NETWORK_ERROR",
                error_type=type(error).__name__,
            )
            raise ModelCallError("Rerank 请求失败（网络或超时）") from error
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.last_call = _telemetry(
                request_id=request_id,
                response_request_id=response_request_id,
                model=self.model,
                document_count=len(documents),
                top_n=top_n,
                started=started,
                estimated_input_tokens=estimated_input_tokens,
                status="INVALID_JSON",
                error_type=type(error).__name__,
            )
            raise ModelResponseError("Rerank 返回结果不是有效 JSON") from error
        try:
            result = _parse_results(data, document_count=len(documents), top_n=top_n)
        except ModelResponseError as error:
            self.last_call = _telemetry(
                request_id=request_id,
                response_request_id=response_request_id,
                model=self.model,
                document_count=len(documents),
                top_n=top_n,
                started=started,
                estimated_input_tokens=estimated_input_tokens,
                status="INVALID_RESPONSE",
                error_type=type(error).__name__,
            )
            raise
        usage = data.get("usage") if isinstance(data, dict) else None
        self.last_call = _telemetry(
            request_id=request_id,
            response_request_id=response_request_id,
            model=self.model,
            document_count=len(documents),
            top_n=top_n,
            started=started,
            estimated_input_tokens=estimated_input_tokens,
            status="ok",
            usage=usage,
        )
        return result


def _header_value(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    value = headers.get(name) if headers is not None else None
    return str(value) if value else None


def _usage_int(usage: object, *names: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _telemetry(
    *,
    request_id: str,
    response_request_id: str | None,
    model: str,
    document_count: int,
    top_n: int,
    started: float,
    estimated_input_tokens: int,
    status: str,
    error_type: str | None = None,
    usage: object = None,
) -> RerankCallTelemetry:
    return RerankCallTelemetry(
        request_id=response_request_id or request_id,
        model=model,
        document_count=document_count,
        top_n=top_n,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=_usage_int(usage, "input_tokens", "prompt_tokens"),
        output_tokens=_usage_int(usage, "output_tokens", "completion_tokens"),
        total_tokens=_usage_int(usage, "total_tokens"),
        estimated_input_tokens=estimated_input_tokens,
        status=status,
        error_type=error_type,
    )


def _rerank_endpoint(base_url: str) -> str:
    """将百炼 Embedding 的 compatible-mode base URL 转成 Rerank API 路径。"""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/compatible-mode/v1"):
        path = path[: -len("/compatible-mode/v1")] + "/compatible-api/v1"
    elif not path.endswith("/compatible-api/v1"):
        # 已经是其他兼容上游时，保留其 base path，不擅自改写供应商路由。
        path = path or "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/reranks", "", ""))


def _parse_results(
    data: object,
    *,
    document_count: int,
    top_n: int,
) -> tuple[RerankResult, ...]:
    if not isinstance(data, dict):
        raise ModelResponseError("Rerank 返回结果必须是 JSON object")
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise ModelResponseError("Rerank 返回结果缺少 results 数组")
    if len(raw_results) != top_n:
        raise ModelResponseError("Rerank 返回数量与 top_n 不一致")

    parsed: list[RerankResult] = []
    seen: set[int] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            raise ModelResponseError("Rerank results 项必须是 JSON object")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ModelResponseError("Rerank index 必须是整数")
        if index < 0 or index >= document_count:
            raise ModelResponseError("Rerank index 超出 documents 范围")
        if index in seen:
            raise ModelResponseError("Rerank 返回了重复 index")
        try:
            relevance_score = float(score)
        except (TypeError, ValueError) as error:
            raise ModelResponseError("Rerank relevance_score 必须是数字") from error
        if not math.isfinite(relevance_score):
            raise ModelResponseError("Rerank relevance_score 必须是有限数字")
        seen.add(index)
        parsed.append(RerankResult(index=index, relevance_score=relevance_score))
    return tuple(parsed)
