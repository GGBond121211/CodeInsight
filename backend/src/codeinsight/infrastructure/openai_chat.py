"""OpenAI-compatible Chat 模型适配器。"""

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
    """解析 JSON 对象，并兼容模型返回的 Markdown 代码围栏。"""
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
            raise ModelResponseError("模型返回内容不是有效 JSON") from None
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise ModelResponseError("模型返回内容不是有效 JSON") from error
    if not isinstance(payload, dict):
        raise ModelResponseError("模型返回内容必须是 JSON 对象")
    return payload


def parse_model_answer(
    content: str,
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> ModelAnswer:
    """将模型返回的 JSON 业务结果解析成不可变领域对象。"""
    payload = _decode_json_object(content)
    if frozenset(payload) != EXPECTED_RESPONSE_KEYS:
        raise ModelResponseError("模型返回结果只能包含 outcome、answer 和 citations")

    outcome = payload.get("outcome")
    answer = payload.get("answer")
    citations = payload.get("citations")
    if outcome not in SUPPORTED_OUTCOMES:
        raise ModelResponseError("模型返回了不支持的 outcome")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelResponseError("模型返回的 answer 不能为空")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise ModelResponseError("模型返回的 citations 必须是 evidence ID 列表")
    if outcome == ANSWERED and not citations:
        raise ModelResponseError("outcome 为 answered 时必须引用证据")
    if outcome == INSUFFICIENT_EVIDENCE and citations:
        raise ModelResponseError("outcome 为 insufficient_evidence 时不能引用证据")

    return ModelAnswer(
        outcome=outcome,
        answer=answer.strip(),
        evidence_ids=tuple(dict.fromkeys(citations)),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OpenAIChatModel:
    """连接一个 OpenAI-compatible Chat Completions 接口的最小适配器。"""

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
            raise ModelConfigurationError("必须配置 CODEINSIGHT_API_KEY")
        if not model:
            raise ModelConfigurationError("必须配置 CODEINSIGHT_MODEL")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=CHAT_TIMEOUT_SECONDS,
            max_retries=0,
        )
        return cls(client=client, model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
        """调用已配置模型，并返回原始内容和用量信息。"""
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
            raise ModelCallError("模型请求失败") from error

        content = response.choices[0].message.content
        if not content:
            raise ModelResponseError("模型返回内容为空")
        usage = response.usage
        return ModelCompletion(
            content=content,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
        """调用已配置模型，并校验回答结果。"""
        completion = self.complete(system_prompt, user_prompt)
        return parse_model_answer(
            completion.content,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
