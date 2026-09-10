"""不重试、不持久化的 OpenAI-compatible Embedding 适配器。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from openai import OpenAI, OpenAIError

from codeinsight.domain.errors import ModelCallError, ModelConfigurationError, ModelResponseError
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.chat_endpoint import configured_chat_base_url

EMBEDDING_BATCH_SIZE = 16


class OpenAIEmbeddingModel:
    """连接一个已配置的多语言 Embedding 服务的最小适配器。"""

    def __init__(self, *, client: OpenAI, model: str, output_type: str | None = None) -> None:
        self._client = client
        self.model = model
        # 部分 Provider 默认只返回 Dense，需要显式请求 dense&sparse 才带稀疏向量。
        # 未配置时保持请求形状不变，避免把参数发给不支持它的端点。
        self.output_type = output_type

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> OpenAIEmbeddingModel:
        source = os.environ if environ is None else environ
        api_key = source.get("CODEINSIGHT_EMBEDDING_API_KEY") or source.get("CODEINSIGHT_API_KEY")
        model = source.get("CODEINSIGHT_EMBEDDING_MODEL")
        base_url = source.get("CODEINSIGHT_EMBEDDING_BASE_URL")
        output_type = (source.get("CODEINSIGHT_EMBEDDING_OUTPUT_TYPE") or "").strip() or None
        if not base_url:
            base_url = configured_chat_base_url(source)
        if not api_key:
            raise ModelConfigurationError(
                "必须配置 CODEINSIGHT_EMBEDDING_API_KEY 或 CODEINSIGHT_API_KEY"
            )
        if not model:
            raise ModelConfigurationError("必须配置 CODEINSIGHT_EMBEDDING_MODEL")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=30.0,
            max_retries=0,
        )
        return cls(client=client, model=model, output_type=output_type)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """为非空文本批次生成向量，并保留服务商用量信息。"""
        if not texts:
            raise ValueError("Embedding 输入必须包含非空文本")
        for text in texts:
            if not text.strip():
                raise ValueError("Embedding 输入必须包含非空文本")
        vectors: list[tuple[float, ...]] = []
        sparse_vectors: list[SparseEmbedding] = []
        sparse_seen = False
        sparse_missing = False
        input_tokens = 0
        usage_available = True
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
            request: dict[str, object] = {"model": self.model, "input": batch}
            if self.output_type is not None:
                request["extra_body"] = {"output_type": self.output_type}
            try:
                response = self._client.embeddings.create(**request)
            except OpenAIError as error:
                code = getattr(error, "code", None) or type(error).__name__
                raise ModelCallError(f"Embedding 请求失败（{code}）") from error

            response_data = response.data or ()

            def data_index(item: object) -> int:
                return item.index

            data = sorted(response_data, key=data_index)
            if len(data) != len(batch):
                raise ModelResponseError("Embedding 返回数量与输入数量不一致")
            for item in data:
                vector_values: list[float] = []
                for value in item.embedding:
                    vector_values.append(float(value))
                vector = tuple(vector_values)
                if not vector:
                    raise ModelResponseError("Embedding 返回结果包含空向量")
                vectors.append(vector)
            batch_sparse = tuple(_parse_sparse_embedding(item) for item in data)
            if any(item is not None for item in batch_sparse):
                sparse_seen = True
                if not all(item is not None for item in batch_sparse):
                    raise ModelResponseError("Embedding 返回的 Dense/Sparse 数量不一致")
                sparse_vectors.extend(item for item in batch_sparse if item is not None)
            else:
                sparse_missing = True
            if sparse_seen and sparse_missing:
                raise ModelResponseError("Embedding 返回的 Dense/Sparse 批次不一致")
            usage = response.usage
            if usage is None:
                usage_available = False
            else:
                input_tokens += usage.total_tokens
        dimensions = len(vectors[0])
        for vector in vectors:
            if len(vector) != dimensions:
                raise ModelResponseError("Embedding 返回向量的维度不一致")
        reported_input_tokens: int | None = input_tokens
        if not usage_available:
            reported_input_tokens = None
        sparse_result = tuple(sparse_vectors) if sparse_seen else None
        return EmbeddingBatch(
            model=self.model,
            vectors=tuple(vectors),
            input_tokens=reported_input_tokens,
            sparse_vectors=sparse_result,
            response_schema_version=(
                "dense-sparse-v1" if sparse_result is not None else "dense-v1"
            ),
        )


def _parse_sparse_embedding(item: object) -> SparseEmbedding | None:
    """读取已登记的 Provider sparse 字段，不根据模型名称猜测能力。"""

    candidate = _value(item, "sparse_embedding")
    if candidate is None:
        candidate = _value(item, "sparse_vector")
    if candidate is None:
        candidate = _value(item, "sparse")
    if candidate is None:
        indices = _value(item, "sparse_indices")
        values = _value(item, "sparse_values")
        if indices is None and values is None:
            return None
        candidate = {"indices": indices, "values": values}

    try:
        pairs = _sparse_pairs(candidate)
        # Provider 可能按权重而不是索引返回；domain 的 SparseEmbedding 要求索引升序。
        pairs.sort(key=lambda pair: pair[0])
        return SparseEmbedding(
            tuple(index for index, _ in pairs),
            tuple(value for _, value in pairs),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelResponseError("Embedding 的 Sparse 结果格式无效") from error


def _sparse_pairs(candidate: object) -> list[tuple[int, float]]:
    """把已登记的两种 Sparse 表示统一成 (index, value) 列表。

    一是 ``{"indices": [...], "values": [...]}``；二是 Provider 实际返回的
    ``[{"index": 9026, "value": 2.88, "token": "..."}, ...]``（阿里云 MaaS）。
    """

    if isinstance(candidate, (list, tuple)):
        pairs: list[tuple[int, float]] = []
        for entry in candidate:
            index = _value(entry, "index")
            value = _value(entry, "value")
            if index is None or value is None:
                raise ModelResponseError("Embedding 的 Sparse 项必须包含 index 和 value")
            pairs.append((int(index), float(value)))
        return pairs

    indices = _value(candidate, "indices")
    values = _value(candidate, "values")
    if not isinstance(indices, (list, tuple)) or not isinstance(values, (list, tuple)):
        raise ModelResponseError("Embedding 的 Sparse 结果必须包含 indices 和 values 数组")
    return [
        (int(index), float(value))
        for index, value in zip(indices, values, strict=True)
    ]


def _value(item: object, name: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)
