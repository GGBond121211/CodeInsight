"""One OpenAI-compatible Embeddings adapter with no retries or persistence."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from openai import OpenAI, OpenAIError

from codeinsight.domain.errors import ModelCallError, ModelConfigurationError, ModelResponseError
from codeinsight.domain.semantic import EmbeddingBatch

EMBEDDING_BATCH_SIZE = 16


class OpenAIEmbeddingModel:
    """Minimal adapter for one configured multilingual embedding endpoint."""

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
        base_url = source.get("CODEINSIGHT_EMBEDDING_BASE_URL") or source.get(
            "CODEINSIGHT_BASE_URL"
        )
        if not api_key:
            raise ModelConfigurationError(
                "CODEINSIGHT_EMBEDDING_API_KEY or CODEINSIGHT_API_KEY is required"
            )
        if not model:
            raise ModelConfigurationError("CODEINSIGHT_EMBEDDING_MODEL is required")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=30.0,
            max_retries=0,
        )
        return cls(client=client, model=model)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed a non-empty batch and preserve provider usage metadata."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty text")
        vectors: list[tuple[float, ...]] = []
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
                raise ModelCallError(f"embedding request failed ({code})") from error

            data = sorted(response.data or (), key=lambda item: item.index)
            if len(data) != len(batch):
                raise ModelResponseError("embedding response count does not match input count")
            for item in data:
                vector = tuple(float(value) for value in item.embedding)
                if not vector:
                    raise ModelResponseError("embedding response contains an empty vector")
                vectors.append(vector)
            usage = response.usage
            if usage is None:
                usage_available = False
            else:
                input_tokens += usage.total_tokens
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise ModelResponseError("embedding response vectors have inconsistent dimensions")
        return EmbeddingBatch(
            model=self.model,
            vectors=tuple(vectors),
            input_tokens=input_tokens if usage_available else None,
        )
