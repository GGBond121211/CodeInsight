"""LangGraph 引用审查和有界答案修订工作流。"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from codeinsight.agent.state import CitationAgentState
from codeinsight.application.answer_repository import map_model_answer
from codeinsight.application.search_repository import (
    build_repository_semantic_index,
    retrieve_subquestion_evidence,
    search_repository,
)
from codeinsight.domain.agent import (
    REVIEW_PASS,
    REVIEW_REVISE,
    AgentEvent,
    AgentRepositoryAnswer,
    CitationReview,
)
from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    PARTIALLY_ANSWERED,
    ModelAnswer,
    ModelCompletion,
    RepositoryAnswer,
    SubQuestionAnswer,
)
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.query_plan import QueryPlan
from codeinsight.domain.retrieval import RankedChunk, SubQuestionEvidence
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.openai_chat import parse_model_answer
from codeinsight.prompts.citation_review import (
    AGENT_PROMPT_VERSION,
    build_review_prompt,
    build_revision_prompt,
    parse_citation_review,
)
from codeinsight.prompts.code_answer import build_answer_prompt
from codeinsight.retrieval.persistent_semantic import embedding_model_id

CompleteModel = Callable[[str, str], ModelCompletion]
SearchRepository = Callable[..., tuple[RankedChunk, ...]]
SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]
MAX_REVISIONS = 5


def _event(state: CitationAgentState, step: str, summary: str) -> tuple[AgentEvent, ...]:
    existing = state.get("events", ())
    return existing + (AgentEvent(len(existing) + 1, step, summary),)


def _usage_update(state: CitationAgentState, completion: ModelCompletion) -> dict[str, int]:
    return {
        "input_tokens": state.get("input_tokens", 0) + (completion.input_tokens or 0),
        "output_tokens": state.get("output_tokens", 0) + (completion.output_tokens or 0),
    }


def _usage_updates(
    state: CitationAgentState, completions: Sequence[ModelCompletion]
) -> dict[str, int]:
    return {
        "input_tokens": state.get("input_tokens", 0)
        + sum(item.input_tokens or 0 for item in completions),
        "output_tokens": state.get("output_tokens", 0)
        + sum(item.output_tokens or 0 for item in completions),
    }


def _parsed_answer(completion: ModelCompletion) -> ModelAnswer:
    return parse_model_answer(
        completion.content,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


def _validate_local_citations(generated: ModelAnswer, evidence_ids: Sequence[str]) -> ModelAnswer:
    supplied = frozenset(evidence_ids)
    unknown = [item for item in generated.evidence_ids if item not in supplied]
    if unknown:
        raise ModelResponseError(f"子问题答案使用了其本地证据组之外的 evidence ID：{unknown[0]}")
    return generated


def _aggregate_outcome(outcomes: Sequence[str]) -> str:
    answered = any(item == ANSWERED for item in outcomes)
    insufficient = any(item == INSUFFICIENT_EVIDENCE for item in outcomes)
    if answered and insufficient:
        return PARTIALLY_ANSWERED
    return ANSWERED if answered else INSUFFICIENT_EVIDENCE


def _combine_subquestion_drafts(plan: QueryPlan, drafts: Sequence[ModelAnswer]) -> ModelAnswer:
    if len(plan.subquestions) != len(drafts):
        raise ValueError("子问题草稿数量必须与 QueryPlan 匹配")
    answer = "\n\n".join(
        f"{index}. {subquestion.question}\n{draft.answer}"
        for index, (subquestion, draft) in enumerate(
            zip(plan.subquestions, drafts, strict=True), start=1
        )
    )
    evidence_ids = tuple(
        dict.fromkeys(evidence_id for draft in drafts for evidence_id in draft.evidence_ids)
    )
    model = next((draft.model for draft in drafts if draft.model), "")
    return ModelAnswer(
        outcome=_aggregate_outcome(tuple(draft.outcome for draft in drafts)),
        answer=answer,
        evidence_ids=evidence_ids,
        model=model,
        input_tokens=sum(draft.input_tokens or 0 for draft in drafts),
        output_tokens=sum(draft.output_tokens or 0 for draft in drafts),
    )


def _map_subquestion_answers(state: CitationAgentState) -> tuple[SubQuestionAnswer, ...]:
    plan = state["query_plan"]
    drafts = state["subquestion_drafts"]
    evidence_groups = state["subquestion_evidence"]
    if not (len(plan.subquestions) == len(drafts) == len(evidence_groups)):
        raise ValueError("QueryPlan、证据和草稿数量必须匹配")
    answers = []
    for subquestion, evidence, draft in zip(
        plan.subquestions, evidence_groups, drafts, strict=True
    ):
        mapped = map_model_answer(
            evidence.results,
            draft,
            retrieval_mode=subquestion.retrieval_mode,
            prompt_version=AGENT_PROMPT_VERSION,
            evidence_ids=evidence.evidence_ids,
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
    return tuple(answers)


def _agent_result(
    state: CitationAgentState,
    generated: ModelAnswer,
    *,
    evidence_ids: tuple[str, ...] | None = None,
    embedding_input_tokens: int = 0,
    summary: str,
) -> AgentRepositoryAnswer:
    input_tokens = state.get("input_tokens", 0)
    output_tokens = state.get("output_tokens", 0)
    subquestions: tuple[SubQuestionAnswer, ...] = ()
    if state.get("query_plan") and state.get("subquestion_drafts"):
        subquestions = _map_subquestion_answers(state)
        generated = _combine_subquestion_drafts(state["query_plan"], state["subquestion_drafts"])
        citations = tuple(
            dict.fromkeys(citation for item in subquestions for citation in item.citations)
        )
        answer = RepositoryAnswer(
            outcome=_aggregate_outcome(tuple(item.outcome for item in subquestions)),
            answer=generated.answer,
            citations=citations,
            retrieval_mode=state["retrieval_mode"],
            model=generated.model or None,
            prompt_version=AGENT_PROMPT_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    else:
        if evidence_ids is not None:
            generated = replace(generated, evidence_ids=evidence_ids)
        answer = map_model_answer(
            state["results"],
            generated,
            retrieval_mode=state["retrieval_mode"],
            prompt_version=AGENT_PROMPT_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    events = _event(state, "finalize", summary)
    return AgentRepositoryAnswer(
        result=answer,
        revisions=state.get("revisions", 0),
        events=events,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_input_tokens=embedding_input_tokens,
        subquestions=subquestions,
    )


def _retrieve_query_plan(
    state: CitationAgentState,
    *,
    search: SearchRepository,
    semantic_embed: SemanticEmbed | None,
    semantic_index,
) -> tuple[
    tuple[RankedChunk, ...],
    tuple[tuple[str, tuple[RankedChunk, ...]], ...],
    tuple[SubQuestionEvidence, ...],
]:
    plan = state["query_plan"]
    unique: dict[tuple[str, int, int], RankedChunk] = {}
    groups: list[tuple[str, tuple[RankedChunk, ...]]] = []
    for subquestion in plan.subquestions:
        retrieved = retrieve_subquestion_evidence(
            state["repository_root"],
            subquestion.question,
            primary_mode=subquestion.retrieval_mode,
            limit=state["limit"],
            semantic_embed=semantic_embed,
            semantic_index=semantic_index,
            search=search,
        )
        groups.append((subquestion.question, retrieved))
        for result in retrieved:
            key = (result.chunk.relative_path, result.chunk.start_line, result.chunk.end_line)
            unique.setdefault(key, result)
    evidence_ids = {key: f"E{index}" for index, key in enumerate(unique, start=1)}
    evidence_groups = tuple(
        SubQuestionEvidence(
            question=subquestion.question,
            retrieval_mode=subquestion.retrieval_mode,
            results=retrieved,
            evidence_ids=tuple(
                evidence_ids[(item.chunk.relative_path, item.chunk.start_line, item.chunk.end_line)]
                for item in retrieved
            ),
        )
        for subquestion, (_, retrieved) in zip(plan.subquestions, groups, strict=True)
    )
    results = tuple(
        RankedChunk(item.chunk, item.score, rank, item.retrieval_reason)
        for rank, item in enumerate(unique.values(), start=1)
    )
    return results, tuple(groups), tuple(evidence_groups)


def run_citation_agent(
    repository_root: str | Path,
    question: str,
    *,
    complete: CompleteModel,
    limit: int = 5,
    retrieval_mode: str = "hybrid",
    search: SearchRepository = search_repository,
    semantic_embed: SemanticEmbed | None = None,
    query_plan: QueryPlan | None = None,
) -> AgentRepositoryAnswer:
    """运行一次检索/起草/审查/修订工作流，最多修订五次。"""
    embedding_input_tokens = 0

    def tracked_semantic_embed(texts: Sequence[str]) -> EmbeddingBatch:
        nonlocal embedding_input_tokens
        if semantic_embed is None:
            raise ValueError("hybrid 检索需要 Embedding 模型")
        batch = semantic_embed(texts)
        embedding_input_tokens += batch.input_tokens or 0
        return batch

    retrieval_embed = tracked_semantic_embed if semantic_embed is not None else None
    semantic_index = None
    if retrieval_embed is not None and (retrieval_mode == "hybrid" or query_plan is not None):
        semantic_index = build_repository_semantic_index(
            repository_root,
            semantic_embed=retrieval_embed,
            semantic_model=embedding_model_id(semantic_embed),
        )

    def retrieve(state: CitationAgentState) -> CitationAgentState:
        if state.get("query_plan"):
            results, groups, evidence_groups = _retrieve_query_plan(
                state,
                search=search,
                semantic_embed=retrieval_embed,
                semantic_index=semantic_index,
            )
        else:
            results = search(
                state["repository_root"],
                state["question"],
                limit=state["limit"],
                retrieval_mode=state["retrieval_mode"],
                semantic_embed=retrieval_embed,
                semantic_index=semantic_index,
            )
            groups = ()
            evidence_groups = ()
        summary = f"已检索 {len(results)} 个证据块。"
        if state.get("query_plan"):
            summary = (
                f"已为 {len(state['query_plan'].subquestions)} 个子问题检索并去重，"
                f"得到 {len(results)} 个证据块。"
            )
        return {
            "results": results,
            "subquestion_results": groups,
            "subquestion_evidence": evidence_groups,
            "events": _event(state, "retrieve", summary),
        }

    def route_retrieval(state: CitationAgentState) -> str:
        if state.get("query_plan"):
            return "draft"
        return "draft" if state["results"] else "no_evidence"

    def finalize_no_evidence(state: CitationAgentState) -> CitationAgentState:
        answer = RepositoryAnswer(
            outcome=INSUFFICIENT_EVIDENCE,
            answer="仓库中没有足够证据回答这个问题。",
            citations=(),
            retrieval_mode=state["retrieval_mode"],
            model=None,
            prompt_version=AGENT_PROMPT_VERSION,
            input_tokens=0,
            output_tokens=0,
        )
        events = _event(state, "finalize", "没有匹配到证据，未调用模型并已停止。")
        return {
            "events": events,
            "agent_result": AgentRepositoryAnswer(
                answer,
                0,
                events,
                0,
                0,
                embedding_input_tokens,
            ),
        }

    def draft(state: CitationAgentState) -> CitationAgentState:
        if state.get("query_plan"):
            completions: list[ModelCompletion] = []
            generated_items: list[ModelAnswer] = []
            accepted: list[bool] = []
            for evidence in state["subquestion_evidence"]:
                if not evidence.results:
                    generated_items.append(
                        ModelAnswer(
                            outcome=INSUFFICIENT_EVIDENCE,
                            answer=("仓库中没有足够证据回答这个子问题。"),
                            evidence_ids=(),
                            model="",
                            input_tokens=0,
                            output_tokens=0,
                        )
                    )
                    accepted.append(True)
                    continue
                system_prompt, user_prompt = build_answer_prompt(
                    evidence.question,
                    evidence.results,
                    evidence_ids=evidence.evidence_ids,
                )
                completion = complete(system_prompt, user_prompt)
                completions.append(completion)
                generated = _validate_local_citations(
                    _parsed_answer(completion), evidence.evidence_ids
                )
                generated_items.append(generated)
                accepted.append(generated.outcome == INSUFFICIENT_EVIDENCE)
            generated = _combine_subquestion_drafts(state["query_plan"], generated_items)
            return {
                "draft": generated,
                "subquestion_drafts": tuple(generated_items),
                "subquestion_reviews": tuple(None for _ in generated_items),
                "subquestion_accepted": tuple(accepted),
                "events": _event(
                    state,
                    "draft",
                    f"已独立起草 {len(generated_items)} 个子问题答案。",
                ),
                **_usage_updates(state, completions),
            }

        system_prompt, user_prompt = build_answer_prompt(state["question"], state["results"])
        completion = complete(system_prompt, user_prompt)
        generated = _parsed_answer(completion)
        return {
            "draft": generated,
            "events": _event(state, "draft", f"已起草一个结果为 {generated.outcome} 的答案。"),
            **_usage_update(state, completion),
        }

    def route_draft(state: CitationAgentState) -> str:
        if state.get("subquestion_drafts"):
            return (
                "finalize_draft"
                if all(
                    item.outcome == INSUFFICIENT_EVIDENCE for item in state["subquestion_drafts"]
                )
                else "review"
            )
        return "finalize_draft" if state["draft"].outcome == INSUFFICIENT_EVIDENCE else "review"

    def review(state: CitationAgentState) -> CitationAgentState:
        if state.get("query_plan") and state.get("subquestion_drafts"):
            completions: list[ModelCompletion] = []
            drafts = list(state["subquestion_drafts"])
            reviews = list(state.get("subquestion_reviews", (None,) * len(drafts)))
            accepted = list(state.get("subquestion_accepted", (False,) * len(drafts)))
            for index, (evidence, draft) in enumerate(
                zip(state["subquestion_evidence"], drafts, strict=True)
            ):
                if accepted[index] or draft.outcome == INSUFFICIENT_EVIDENCE:
                    accepted[index] = True
                    continue
                system_prompt, user_prompt = build_review_prompt(
                    evidence.question,
                    evidence.results,
                    draft,
                )
                completion = complete(system_prompt, user_prompt)
                completions.append(completion)
                result = parse_citation_review(completion.content, frozenset(evidence.evidence_ids))
                reviews[index] = result
                if result.verdict == REVIEW_PASS:
                    drafts[index] = replace(draft, evidence_ids=result.supported_evidence_ids)
                    accepted[index] = True
            pending = sum(not item for item in accepted)
            verdict = REVIEW_PASS if pending == 0 else REVIEW_REVISE
            aggregate_review = CitationReview(
                verdict=verdict,
                supported_evidence_ids=tuple(
                    dict.fromkeys(
                        evidence_id for draft in drafts for evidence_id in draft.evidence_ids
                    )
                ),
                feedback=(
                    "所有子问题答案都通过了独立引用审查。"
                    if pending == 0
                    else f"有 {pending} 个子问题答案需要修订。"
                ),
            )
            return {
                "draft": _combine_subquestion_drafts(state["query_plan"], drafts),
                "subquestion_drafts": tuple(drafts),
                "subquestion_reviews": tuple(reviews),
                "subquestion_accepted": tuple(accepted),
                "review": aggregate_review,
                "events": _event(
                    state,
                    "review",
                    f"已独立审查子问题；有 {pending} 个需要修订。",
                ),
                **_usage_updates(state, completions),
            }

        system_prompt, user_prompt = build_review_prompt(
            state["question"],
            state["results"],
            state["draft"],
        )
        completion = complete(system_prompt, user_prompt)
        supplied_ids = frozenset(f"E{index}" for index, _ in enumerate(state["results"], start=1))
        result = parse_citation_review(completion.content, supplied_ids)
        return {
            "review": result,
            "events": _event(state, "review", f"引用审查结论：{result.verdict}。"),
            **_usage_update(state, completion),
        }

    def route_review(state: CitationAgentState) -> str:
        if state["review"].verdict == REVIEW_PASS:
            return "finalize_reviewed"
        return (
            "revise"
            if state.get("revisions", 0) < state.get("max_revisions", MAX_REVISIONS)
            else "finalize_revised"
        )

    def revise(state: CitationAgentState) -> CitationAgentState:
        if state.get("query_plan") and state.get("subquestion_drafts"):
            completions: list[ModelCompletion] = []
            drafts = list(state["subquestion_drafts"])
            accepted = list(state["subquestion_accepted"])
            for index, (evidence, draft, review_result) in enumerate(
                zip(
                    state["subquestion_evidence"],
                    drafts,
                    state["subquestion_reviews"],
                    strict=True,
                )
            ):
                if accepted[index]:
                    continue
                if review_result is None:
                    raise ValueError("待修订的子问题必须有审查反馈")
                system_prompt, user_prompt = build_revision_prompt(
                    evidence.question,
                    evidence.results,
                    draft,
                    review_result,
                )
                completion = complete(system_prompt, user_prompt)
                completions.append(completion)
                drafts[index] = _validate_local_citations(
                    _parsed_answer(completion), evidence.evidence_ids
                )
                accepted[index] = drafts[index].outcome == INSUFFICIENT_EVIDENCE
            generated = _combine_subquestion_drafts(state["query_plan"], drafts)
            revision = state.get("revisions", 0) + 1
            return {
                "draft": generated,
                "subquestion_drafts": tuple(drafts),
                "subquestion_accepted": tuple(accepted),
                "revised": generated,
                "revisions": revision,
                "events": _event(
                    state,
                    "revise",
                    f"只修订了未通过的子问题（第 {revision}/"
                    f"{state.get('max_revisions', MAX_REVISIONS)} 次）。",
                ),
                **_usage_updates(state, completions),
            }

        system_prompt, user_prompt = build_revision_prompt(
            state["question"],
            state["results"],
            state["draft"],
            state["review"],
        )
        completion = complete(system_prompt, user_prompt)
        generated = _parsed_answer(completion)
        return {
            "draft": generated,
            "revised": generated,
            "revisions": state.get("revisions", 0) + 1,
            "events": _event(
                state,
                "revise",
                f"根据审查反馈修订答案（第 {state.get('revisions', 0) + 1}/"
                f"{state.get('max_revisions', MAX_REVISIONS)} 次）。",
            ),
            **_usage_update(state, completion),
        }

    def finalize_draft(state: CitationAgentState) -> CitationAgentState:
        return {
            "agent_result": _agent_result(
                state,
                state["draft"],
                embedding_input_tokens=embedding_input_tokens,
                summary="证据不足的草稿无需引用审查，已接受。",
            )
        }

    def finalize_reviewed(state: CitationAgentState) -> CitationAgentState:
        return {
            "agent_result": _agent_result(
                state,
                state["draft"],
                evidence_ids=state["review"].supported_evidence_ids,
                embedding_input_tokens=embedding_input_tokens,
                summary="已接受通过审查的草稿。",
            )
        }

    def finalize_revised(state: CitationAgentState) -> CitationAgentState:
        result_state = state
        if state.get("subquestion_drafts"):
            drafts = tuple(
                replace(draft, evidence_ids=review_result.supported_evidence_ids)
                if not accepted and review_result is not None
                else draft
                for draft, review_result, accepted in zip(
                    state["subquestion_drafts"],
                    state["subquestion_reviews"],
                    state["subquestion_accepted"],
                    strict=True,
                )
            )
            result_state = {**state, "subquestion_drafts": drafts}
        evidence_ids = (
            ()
            if state["draft"].outcome == INSUFFICIENT_EVIDENCE
            else state["review"].supported_evidence_ids
        )
        return {
            "agent_result": _agent_result(
                result_state,
                state["draft"],
                evidence_ids=evidence_ids,
                embedding_input_tokens=embedding_input_tokens,
                summary=(
                    "审查者接受修订后的答案，流程已完成。"
                    if state["review"].verdict == REVIEW_PASS
                    else "达到五次修订上限，已完成当前最佳答案。"
                ),
            )
        }

    graph = StateGraph(CitationAgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("no_evidence", finalize_no_evidence)
    graph.add_node("draft", draft)
    graph.add_node("review", review)
    graph.add_node("revise", revise)
    graph.add_node("finalize_draft", finalize_draft)
    graph.add_node("finalize_reviewed", finalize_reviewed)
    graph.add_node("finalize_revised", finalize_revised)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_retrieval, {"draft": "draft", "no_evidence": "no_evidence"}
    )
    graph.add_edge("no_evidence", END)
    graph.add_conditional_edges(
        "draft", route_draft, {"finalize_draft": "finalize_draft", "review": "review"}
    )
    graph.add_edge("finalize_draft", END)
    graph.add_conditional_edges(
        "review",
        route_review,
        {
            "finalize_reviewed": "finalize_reviewed",
            "revise": "revise",
            "finalize_revised": "finalize_revised",
        },
    )
    graph.add_edge("finalize_reviewed", END)
    graph.add_conditional_edges(
        "revise",
        lambda state: (
            "finalize_revised" if state["revised"].outcome == INSUFFICIENT_EVIDENCE else "review"
        ),
        {"finalize_revised": "finalize_revised", "review": "review"},
    )
    graph.add_edge("finalize_revised", END)

    final_state = graph.compile().invoke(
        {
            "repository_root": repository_root,
            "question": question,
            "limit": limit,
            "retrieval_mode": retrieval_mode,
            "query_plan": query_plan,
            "route_reason": "query_plan" if query_plan else "explicit_entry",
            "router_confidence": query_plan.confidence if query_plan else 1.0,
            "subquestion_results": (),
            "subquestion_evidence": (),
            "revisions": 0,
            "max_revisions": MAX_REVISIONS,
            "events": (),
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )
    return final_state["agent_result"]
