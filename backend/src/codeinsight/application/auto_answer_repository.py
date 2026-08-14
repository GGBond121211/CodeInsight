"""Multi-question Smart Answer evidence orchestration."""

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
        answer="The repository does not contain enough evidence to answer this subquestion.",
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
    """Retrieve and answer each public subquestion with local evidence coverage."""
    plan: QueryPlan = router_result.plan
    embedding_input_tokens = 0

    def tracked_semantic_embed(texts: Sequence[str]) -> EmbeddingBatch:
        nonlocal embedding_input_tokens
        if semantic_embed is None:
            raise ValueError("hybrid retrieval requires an embedding model")
        batch = semantic_embed(texts)
        embedding_input_tokens += batch.input_tokens or 0
        return batch

    subquestions = plan.subquestions[:MAX_SUBQUESTIONS]
    if plan.execution_route == "insufficient" or not subquestions:
        return AutoAnswer(
            outcome=INSUFFICIENT_EVIDENCE,
            answer="The question could not be mapped to a repository query.",
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
            events=(
                AutoAnswerEvent(
                    1, "route", "Stopped because the router produced no executable subquestion."
                ),
            ),
        )

    if semantic_embed is None:
        raise ValueError("normal answer retrieval requires an embedding model")
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
    covered_subquestions = sum(evidence.covered for _, evidence in per_subquestion)
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

    answered = [item for item in answers if item.outcome == ANSWERED]
    insufficient = [item for item in answers if item.outcome == INSUFFICIENT_EVIDENCE]
    if answered and insufficient:
        outcome = PARTIALLY_ANSWERED
    elif answered:
        outcome = ANSWERED
    else:
        outcome = INSUFFICIENT_EVIDENCE
    sections = [f"{index}. {item.question}\n{item.answer}" for index, item in enumerate(answers, 1)]
    citations = tuple(
        citation
        for index, item in enumerate(answers)
        for citation in item.citations
        if citation
        not in tuple(previous for prior in answers[:index] for previous in prior.citations)
    )
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
                f"Prepared {len(answers)} public subquestions on the linear route.",
            ),
            AutoAnswerEvent(
                2,
                "finalize",
                f"Completed {outcome}; local evidence covered "
                f"{covered_subquestions}/{len(answers)} subquestions.",
            ),
        ),
    )
