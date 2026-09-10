"""结构化 LLM Query Router，以及确定性的 Linear 回退方案。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.errors import ModelCallError, ModelResponseError
from codeinsight.domain.query_plan import (
    SUPPORTED_EXECUTION_ROUTES,
    SUPPORTED_LANGUAGES,
    QueryPlan,
    SubQuestion,
)
from codeinsight.prompts.query_router import PROMPT_VERSION, build_router_prompt

MIN_ROUTER_CONFIDENCE = 0.70
ROUTER_KEYS = frozenset(
    {"language", "normalized_question", "subquestions", "execution_route", "confidence"}
)
SUBQUESTION_KEYS = frozenset({"question", "intent", "retrieval_mode"})
SUPPORTED_INTENTS = frozenset(
    {"symbol_lookup", "call_flow", "data_flow", "implementation", "boundary", "semantic", "unknown"}
)
ROUTER_RETRIEVAL_MODES = frozenset({"hybrid", "dense", "sparse"})
_LANGUAGE_ALIASES = {
    "zh-en": "mixed",
    "en-zh": "mixed",
    "chinese": "zh",
    "english": "en",
}
_INTENT_ALIASES = {
    "call_graph": "call_flow",
    "cross_file_call": "call_flow",
    "dataflow": "data_flow",
    "business_logic": "implementation",
}
_RETRIEVAL_ALIASES = {
    "bm25": "hybrid",
    "semantic": "dense",
}


@dataclass(frozen=True)
class QueryRouterResult:
    plan: QueryPlan
    used_fallback: bool
    fallback_reason: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    elapsed_milliseconds: float
    prompt_version: str = PROMPT_VERSION


def fallback_plan(question: str, reason: str) -> QueryPlan:
    """构造 2.1 的 Dense/Sparse + Linear 回退计划。"""
    subquestion = SubQuestion(question, "unknown", "hybrid")
    return QueryPlan(
        original_question=question,
        language="unknown",
        normalized_question=question,
        subquestions=(subquestion,),
        retrieval_modes=("hybrid",),
        execution_route="linear",
        confidence=0.0,
        fallback_reason=reason,
    )


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelResponseError(f"Router 字段 {key} 必须是非空字符串")
    return value.strip()


def _decode_json_object(content: str) -> dict[str, Any]:
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
            raise ModelResponseError("Router 返回内容不是有效 JSON") from None
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise ModelResponseError("Router 返回内容不是有效 JSON") from error
    if not isinstance(payload, dict):
        raise ModelResponseError("Router 返回内容必须是 JSON 对象")
    return payload


def parse_query_plan(original_question: str, content: str) -> QueryPlan:
    """解析并严格校验 Router 返回的 JSON。"""
    payload = _decode_json_object(content)
    if frozenset(payload) != ROUTER_KEYS:
        raise ModelResponseError("Router 返回的 JSON 结构不受支持")

    language = _require_string(payload, "language")
    language = _LANGUAGE_ALIASES.get(language.lower(), language.lower())
    if language not in SUPPORTED_LANGUAGES:
        raise ModelResponseError("Router 返回了不支持的语言")
    normalized_question = _require_string(payload, "normalized_question")
    execution_route = _require_string(payload, "execution_route")
    if execution_route not in SUPPORTED_EXECUTION_ROUTES:
        raise ModelResponseError("Router 返回了不支持的执行路线")
    confidence = payload.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence.strip())
        except ValueError as error:
            raise ModelResponseError("Router 的 confidence 必须是数字") from error
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ModelResponseError("Router 的 confidence 必须是数字")

    raw_subquestions = payload.get("subquestions")
    if not isinstance(raw_subquestions, list):
        raise ModelResponseError("Router 的 subquestions 必须是列表")
    subquestions: list[SubQuestion] = []
    for raw in raw_subquestions:
        if not isinstance(raw, dict) or frozenset(raw) != SUBQUESTION_KEYS:
            raise ModelResponseError("Router 子问题的 JSON 结构不受支持")
        question = _require_string(raw, "question")
        intent = _require_string(raw, "intent")
        intent = _INTENT_ALIASES.get(intent.lower(), intent.lower())
        if intent not in SUPPORTED_INTENTS:
            raise ModelResponseError("Router 子问题的 intent 不受支持")
        retrieval_mode = _require_string(raw, "retrieval_mode")
        retrieval_mode = _RETRIEVAL_ALIASES.get(retrieval_mode.lower(), retrieval_mode.lower())
        if retrieval_mode not in ROUTER_RETRIEVAL_MODES:
            raise ModelResponseError("Router 子问题的 retrieval_mode 不受支持")
        subquestions.append(SubQuestion(question, intent, retrieval_mode))

    retrieval_mode_list: list[str] = []
    for item in subquestions:
        if item.retrieval_mode not in retrieval_mode_list:
            retrieval_mode_list.append(item.retrieval_mode)
    retrieval_modes = tuple(retrieval_mode_list)
    return QueryPlan(
        original_question=original_question,
        language=language,
        normalized_question=normalized_question,
        subquestions=tuple(subquestions),
        retrieval_modes=retrieval_modes,
        execution_route=execution_route,
        confidence=float(confidence),
    )


def route_question(
    question: str,
    *,
    complete,
    min_confidence: float = MIN_ROUTER_CONFIDENCE,
) -> QueryRouterResult:
    """调用一次 Router；失败时返回安全的 Linear 回退计划。"""
    started = perf_counter()
    input_tokens = 0
    output_tokens = 0
    model: str | None = None
    try:
        system_prompt, user_prompt = build_router_prompt(question)
        completion: ModelCompletion = complete(system_prompt, user_prompt)
        model = completion.model
        input_tokens = completion.input_tokens or 0
        output_tokens = completion.output_tokens or 0
        plan = parse_query_plan(question, completion.content)
        if plan.confidence < min_confidence:
            raise ModelResponseError("Router 的 confidence 低于配置阈值")
        return QueryRouterResult(
            plan=plan,
            used_fallback=False,
            fallback_reason=None,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_milliseconds=(perf_counter() - started) * 1000,
        )
    except (ModelCallError, ModelResponseError, ValueError, TypeError):
        reason = "router_call_failed" if model is None else "router_invalid_or_low_confidence"
        return QueryRouterResult(
            plan=fallback_plan(question, reason),
            used_fallback=True,
            fallback_reason=reason,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_milliseconds=(perf_counter() - started) * 1000,
        )
