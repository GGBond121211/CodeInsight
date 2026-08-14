"""One OpenAI-compatible chat model adapter."""

import json
import os
from collections.abc import Mapping

from openai import OpenAI, OpenAIError

from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED_OUTCOMES,
    ModelAnswer,
    ModelCompletion,
)
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)

EXPECTED_RESPONSE_KEYS = frozenset({"outcome", "answer", "citations"})
CHAT_TIMEOUT_SECONDS = 60.0


def _decode_json_object(content: str) -> dict:
    """Decode a JSON object while tolerating a markdown wrapper from a model."""
    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ModelResponseError("model response is not valid JSON") from None
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise ModelResponseError("model response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ModelResponseError("model response must be a JSON object")
    return payload


def parse_model_answer(
    content: str,
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> ModelAnswer:
    """Parse the model's JSON business response into an immutable value."""
    payload = _decode_json_object(content)
    if frozenset(payload) != EXPECTED_RESPONSE_KEYS:
        raise ModelResponseError("model response must contain only outcome, answer, and citations")

    outcome = payload.get("outcome")
    answer = payload.get("answer")
    citations = payload.get("citations")
    if outcome not in SUPPORTED_OUTCOMES:
        raise ModelResponseError("model response has an unsupported outcome")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelResponseError("model response answer must be non-empty")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise ModelResponseError("model response citations must be a list of evidence IDs")
    if outcome == ANSWERED and not citations:
        raise ModelResponseError("answered model response must cite evidence")
    if outcome == INSUFFICIENT_EVIDENCE and citations:
        raise ModelResponseError("insufficient model response cannot cite evidence")

    return ModelAnswer(
        outcome=outcome,
        answer=answer.strip(),
        evidence_ids=tuple(dict.fromkeys(citations)),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OpenAIChatModel:
    """Minimal adapter for one OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        response_format: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._response_format = response_format

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "OpenAIChatModel":
        source = os.environ if environ is None else environ
        api_key = source.get("CODEINSIGHT_API_KEY")
        model = source.get("CODEINSIGHT_MODEL")
        base_url = source.get("CODEINSIGHT_BASE_URL")
        if not api_key:
            raise ModelConfigurationError("CODEINSIGHT_API_KEY is required")
        if not model:
            raise ModelConfigurationError("CODEINSIGHT_MODEL is required")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=CHAT_TIMEOUT_SECONDS,
            max_retries=0,
        )
        return cls(client=client, model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
        """Call the configured model and return raw content with usage."""
        try:
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            }
            if self._response_format is not None:
                request["response_format"] = self._response_format
            response = self._client.chat.completions.create(
                **request,
            )
        except OpenAIError as error:
            raise ModelCallError("model request failed") from error

        content = response.choices[0].message.content
        if not content:
            raise ModelResponseError("model response content is empty")
        usage = response.usage
        return ModelCompletion(
            content=content,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
        """Call the configured model and validate its answer response."""
        completion = self.complete(system_prompt, user_prompt)
        return parse_model_answer(
            completion.content,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
