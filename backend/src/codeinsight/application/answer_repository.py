"""基于检索到的源码证据回答仓库问题。"""

from collections.abc import Callable, Sequence
from pathlib import Path

from codeinsight.application.search_repository import search_repository
from codeinsight.domain.answer import (
    INSUFFICIENT_EVIDENCE,
    AnswerCitation,
    ModelAnswer,
    RepositoryAnswer,
)
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.reranker import Reranker
from codeinsight.prompts.code_answer import PROMPT_VERSION, build_answer_prompt

GenerateAnswer = Callable[[str, str], ModelAnswer]
SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]


def map_model_answer(
    results: Sequence[RankedChunk],
    generated: ModelAnswer,
    *,
    retrieval_mode: str,
    prompt_version: str,
    evidence_ids: Sequence[str] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> RepositoryAnswer:
    """把模型返回的 evidence ID 映射为可信的仓库路径和行号范围。"""
    if evidence_ids:
        identifiers = tuple(evidence_ids)
    else:
        generated_identifiers: list[str] = []
        for index in range(1, len(results) + 1):
            generated_identifiers.append(f"E{index}")
        identifiers = tuple(generated_identifiers)
    if len(identifiers) != len(results):
        raise ModelResponseError("evidence ID 数量必须与结果数量一致")
    evidence: dict[str, object] = {}
    for evidence_id, result in zip(identifiers, results, strict=True):
        evidence[evidence_id] = result.chunk

    unknown: list[str] = []
    for item in generated.evidence_ids:
        if item not in evidence:
            unknown.append(item)
    if unknown:
        raise ModelResponseError(f"模型引用了未知的 evidence ID：{unknown[0]}")

    citation_list: list[AnswerCitation] = []
    for evidence_id in generated.evidence_ids:
        chunk = evidence[evidence_id]
        citation_list.append(
            AnswerCitation(
                evidence_id=evidence_id,
                relative_path=chunk.relative_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
        )
    citations = tuple(citation_list)
    return RepositoryAnswer(
        outcome=generated.outcome,
        answer=generated.answer,
        citations=citations,
        retrieval_mode=retrieval_mode,
        model=generated.model,
        prompt_version=prompt_version,
        input_tokens=generated.input_tokens if input_tokens is None else input_tokens,
        output_tokens=generated.output_tokens if output_tokens is None else output_tokens,
    )


def answer_repository(
    root: str | Path,
    question: str,
    *,
    generate: GenerateAnswer,
    limit: int = 5,
    chunk_max_lines: int = 80,
    retrieval_mode: str = "hybrid",
    semantic_embed: SemanticEmbed | None = None,
    reranker: Reranker | None = None,
) -> RepositoryAnswer:
    """只使用检索到的源码块回答一个仓库问题。"""
    results = search_repository(
        root,
        question,
        limit=limit,
        chunk_max_lines=chunk_max_lines,
        retrieval_mode=retrieval_mode,
        semantic_embed=semantic_embed,
        reranker=reranker,
    )
    if not results:
        return RepositoryAnswer(
            outcome=INSUFFICIENT_EVIDENCE,
            answer="仓库中没有足够证据回答这个问题。",
            citations=(),
            retrieval_mode=retrieval_mode,
            model=None,
            prompt_version=PROMPT_VERSION,
            input_tokens=None,
            output_tokens=None,
        )

    system_prompt, user_prompt = build_answer_prompt(question, results)
    generated = generate(system_prompt, user_prompt)
    return map_model_answer(
        results,
        generated,
        retrieval_mode=retrieval_mode,
        prompt_version=PROMPT_VERSION,
    )
