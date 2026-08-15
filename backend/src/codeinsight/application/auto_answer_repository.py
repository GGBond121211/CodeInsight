"""多问题 Smart Answer 的证据编排。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from codeinsight.application.answer_repository import map_model_answer
from codeinsight.application.query_router import QueryRouterResult
from codeinsight.application.search_repository import (
    build_repository_semantic_index,
    retrieve_subquestion_evidence,
    search_repository,
)
from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    PARTIALLY_ANSWERED,
    AutoAnswer,
    AutoAnswerEvent,
    ModelAnswer,
    SubQuestionAnswer,
)
from codeinsight.domain.query_plan import QueryPlan, SubQuestion
from codeinsight.domain.retrieval import RankedChunk, SubQuestionEvidence
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.prompts.code_answer import PROMPT_VERSION, build_answer_prompt
from codeinsight.retrieval.persistent_semantic import embedding_model_id

GenerateAnswer = Callable[[str, str], ModelAnswer]
SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]
MAX_SUBQUESTIONS = 8


def _chunk_key(result: RankedChunk) -> tuple[str, int, int]:
    return (
        result.chunk.relative_path,
        result.chunk.start_line,
        result.chunk.end_line,
    )


def _empty_subquestion(question, intent, retrieval_mode) -> SubQuestionAnswer:
    return SubQuestionAnswer(
        question=question,
        intent=intent,
        retrieval_mode=retrieval_mode,
        outcome=INSUFFICIENT_EVIDENCE,
        answer="仓库中没有足够证据回答这个子问题。",
        citations=(),
    )


def auto_answer_repository(
    root: str | Path,
    *,
    router_result: QueryRouterResult,
    generate: GenerateAnswer,
    limit: int = 5,
    chunk_max_lines: int = 80,
    semantic_embed: SemanticEmbed | None = None,
) -> AutoAnswer:
    """分别检索并回答每个公开子问题，同时记录本地证据覆盖情况。"""
    plan: QueryPlan = router_result.plan
    embedding_input_tokens = 0

    def tracked_semantic_embed(texts: Sequence[str]) -> EmbeddingBatch:
        nonlocal embedding_input_tokens
        if semantic_embed is None:
            raise ValueError("hybrid 检索需要 Embedding 模型")
        batch = semantic_embed(texts)
        embedding_input_tokens += batch.input_tokens or 0
        return batch

    subquestions = plan.subquestions[:MAX_SUBQUESTIONS]
    if plan.execution_route == "insufficient" or not subquestions:
        return AutoAnswer(
            outcome=INSUFFICIENT_EVIDENCE,
            answer="无法将这个问题映射到仓库查询。",
            citations=(),
            retrieval_mode="auto",
            model=None,
            prompt_version=PROMPT_VERSION,
            input_tokens=0,
            output_tokens=0,
            plan=plan,
            subquestions=(),
            router_model=router_result.model,
            router_input_tokens=router_result.input_tokens,
            router_output_tokens=router_result.output_tokens,
            router_elapsed_milliseconds=router_result.elapsed_milliseconds,
            embedding_input_tokens=embedding_input_tokens,
            fallback_reason=router_result.fallback_reason,
            events=(AutoAnswerEvent(1, "route", "Router 没有生成可执行的子问题，已停止。"),),
        )

    if semantic_embed is None:
        raise ValueError("普通回答检索需要 Embedding 模型")
    retrieval_embed = tracked_semantic_embed
    semantic_index = build_repository_semantic_index(
        root,
        chunk_max_lines=chunk_max_lines,
        semantic_embed=retrieval_embed,
        semantic_model=embedding_model_id(semantic_embed),
    )

    global_ids: dict[tuple[str, int, int], str] = {}
    per_subquestion: list[tuple[SubQuestion, SubQuestionEvidence]] = []
    for subquestion in subquestions:
        retrieved = retrieve_subquestion_evidence(
            root,
            subquestion.question,
            primary_mode=subquestion.retrieval_mode,
            limit=limit,
            chunk_max_lines=chunk_max_lines,
            semantic_embed=retrieval_embed,
            semantic_index=semantic_index,
            search=search_repository,
        )
        selected: list[RankedChunk] = []
        identifiers: list[str] = []
        for result in retrieved:
            key = _chunk_key(result)
            if key not in global_ids:
                global_ids[key] = f"E{len(global_ids) + 1}"
            selected.append(result)
            identifiers.append(global_ids[key])
        per_subquestion.append(
            (
                subquestion,
                SubQuestionEvidence(
                    question=subquestion.question,
                    retrieval_mode=subquestion.retrieval_mode,
                    results=tuple(selected),
                    evidence_ids=tuple(identifiers),
                ),
            )
        )

    answers: list[SubQuestionAnswer] = []
    answer_models: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    covered_subquestions = 0
    for _, evidence in per_subquestion:
        covered_subquestions += evidence.covered
    for subquestion, evidence in per_subquestion:
        results = evidence.results
        identifiers = evidence.evidence_ids
        if not results:
            answers.append(
                _empty_subquestion(
                    subquestion.question,
                    subquestion.intent,
                    subquestion.retrieval_mode,
                )
            )
            continue
        system_prompt, user_prompt = build_answer_prompt(
            subquestion.question,
            results,
            evidence_ids=identifiers,
        )
        generated = generate(system_prompt, user_prompt)
        total_input_tokens += generated.input_tokens or 0
        total_output_tokens += generated.output_tokens or 0
        if generated.model:
            answer_models.append(generated.model)
        mapped = map_model_answer(
            results,
            generated,
            retrieval_mode=subquestion.retrieval_mode,
            prompt_version=PROMPT_VERSION,
            evidence_ids=identifiers,
        )
        answers.append(
            SubQuestionAnswer(
                question=subquestion.question,
                intent=subquestion.intent,
                retrieval_mode=subquestion.retrieval_mode,
                outcome=mapped.outcome,
                answer=mapped.answer,
                citations=mapped.citations,
            )
        )

    answered: list[SubQuestionAnswer] = []
    insufficient: list[SubQuestionAnswer] = []
    for item in answers:
        if item.outcome == ANSWERED:
            answered.append(item)
        if item.outcome == INSUFFICIENT_EVIDENCE:
            insufficient.append(item)
    if answered and insufficient:
        outcome = PARTIALLY_ANSWERED
    elif answered:
        outcome = ANSWERED
    else:
        outcome = INSUFFICIENT_EVIDENCE
    sections: list[str] = []
    for index, item in enumerate(answers, 1):
        sections.append(f"{index}. {item.question}\n{item.answer}")

    citation_list: list[object] = []
    for index, item in enumerate(answers):
        previous_citations: list[object] = []
        for prior in answers[:index]:
            for previous in prior.citations:
                previous_citations.append(previous)
        for citation in item.citations:
            if citation not in previous_citations:
                citation_list.append(citation)
    citations = tuple(citation_list)
    return AutoAnswer(
        outcome=outcome,
        answer="\n\n".join(sections),
        citations=citations,
        retrieval_mode="auto",
        model=answer_models[0] if answer_models else None,
        prompt_version=PROMPT_VERSION,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        embedding_input_tokens=embedding_input_tokens,
        plan=plan,
        subquestions=tuple(answers),
        router_model=router_result.model,
        router_input_tokens=router_result.input_tokens,
        router_output_tokens=router_result.output_tokens,
        router_elapsed_milliseconds=router_result.elapsed_milliseconds,
        fallback_reason=router_result.fallback_reason,
        events=(
            AutoAnswerEvent(
                1,
                "route",
                f"已在线性路线准备 {len(answers)} 个公开子问题。",
            ),
            AutoAnswerEvent(
                2,
                "finalize",
                f"已完成，结果为 {outcome}；本地证据覆盖 "
                f"{covered_subquestions}/{len(answers)} 个子问题。",
            ),
        ),
    )
