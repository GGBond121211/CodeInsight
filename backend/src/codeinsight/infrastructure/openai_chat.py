"""OpenAI-compatible Chat 模型适配器。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError

from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse
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

if TYPE_CHECKING:
    from codeinsight.application.context_assembler import ContextAssembler

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
    citations_are_strings = True
    if isinstance(citations, list):
        for item in citations:
            if not isinstance(item, str):
                citations_are_strings = False
                break
    else:
        citations_are_strings = False
    if not citations_are_strings:
        raise ModelResponseError("模型返回的 citations 必须是 evidence ID 列表")
    if outcome == ANSWERED and not citations:
        raise ModelResponseError("outcome 为 answered 时必须引用证据")
    if outcome == INSUFFICIENT_EVIDENCE and citations:
        raise ModelResponseError("outcome 为 insufficient_evidence 时不能引用证据")

    unique_citations: list[str] = []
    for citation in citations:
        if citation not in unique_citations:
            unique_citations.append(citation)

    return ModelAnswer(
        outcome=outcome,
        answer=answer.strip(),
        evidence_ids=tuple(unique_citations),
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
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._response_format = response_format
        self._context_assembler = context_assembler

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        context_assembler: ContextAssembler | None = None,
    ) -> OpenAIChatModel:
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
        return cls(client=client, model=model, context_assembler=context_assembler)

    def complete(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
        """调用已配置模型，并返回原始内容和用量信息。"""
        try:
            request_user_prompt = user_prompt
            estimated_input_tokens: int | None = None
            if self._context_assembler is not None:
                from codeinsight.application.context_assembler import ContextRequest

                assembly = self._context_assembler.assemble(
                    ContextRequest(
                        system_safety=system_prompt,
                        user_goal="",
                        user_code_task=user_prompt,
                        model_version=self.model,
                        strategy_version="context-assembler-v1",
                    )
                )
                request_user_prompt = assembly.user_text
                estimated_input_tokens = assembly.fitted.estimate.input_tokens
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request_user_prompt},
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
            estimated_input_tokens=estimated_input_tokens,
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

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
    ) -> ToolModelResponse:
        """调用 Chat Completions 的原生 ``tools`` 字段。

        只读取 SDK 返回的 ``message.tool_calls``。如果模型把工具调用写成
        普通文本 JSON，这里会把它当普通最终文本，不会猜测并执行。
        """
        request = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": 0,
        }
        try:
            response = self._client.chat.completions.create(**request)
        except OpenAIError as error:
            raise ModelCallError("模型请求失败") from error
        message = response.choices[0].message
        normalized_calls: list[ToolCall] = []
        for item in getattr(message, "tool_calls", None) or ():
            function = getattr(item, "function", None)
            if function is None:
                raise ModelResponseError("原生 tool_call 缺少 function")
            try:
                arguments = json.loads(getattr(function, "arguments", "{}"))
            except json.JSONDecodeError as error:
                raise ModelResponseError("原生 tool_call 的 arguments 不是有效 JSON") from error
            if not isinstance(arguments, dict):
                raise ModelResponseError("原生 tool_call 的 arguments 必须是 JSON object")
            normalized_calls.append(
                ToolCall(
                    str(getattr(item, "id", "")),
                    str(getattr(function, "name", "")),
                    arguments,
                )
            )
        usage = response.usage
        return ToolModelResponse(
            content=getattr(message, "content", None),
            tool_calls=tuple(normalized_calls),
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
