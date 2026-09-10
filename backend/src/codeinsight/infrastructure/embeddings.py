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

    def __init__(self, *, client: OpenAI, model: str) -> None:
        self._client = client
        self.model = model

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> OpenAIEmbeddingModel:
        source = os.environ if environ is None else environ
        api_key = source.get("CODEINSIGHT_EMBEDDING_API_KEY") or source.get("CODEINSIGHT_API_KEY")
        model = source.get("CODEINSIGHT_EMBEDDING_MODEL")
        base_url = source.get("CODEINSIGHT_EMBEDDING_BASE_URL")
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
        return cls(client=client, model=model)

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
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
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

    indices = _value(candidate, "indices")
    values = _value(candidate, "values")
    if not isinstance(indices, (list, tuple)) or not isinstance(values, (list, tuple)):
        raise ModelResponseError("Embedding 的 Sparse 结果必须包含 indices 和 values 数组")
    try:
        return SparseEmbedding(
            tuple(int(index) for index in indices),
            tuple(float(value) for value in values),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelResponseError("Embedding 的 Sparse 结果格式无效") from error


def _value(item: object, name: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)
