"""引用审查和答案修订的版本化 Prompt 与解析器。"""

import json
from collections.abc import Sequence

from codeinsight.domain.agent import SUPPORTED_REVIEW_VERDICTS, CitationReview
from codeinsight.domain.answer import ModelAnswer
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.prompts.code_answer import grouped_evidence_blocks

CITATION_REVIEW_VERSION = "citation-review-v2"
AGENT_PROMPT_VERSION = "citation-agent-v3"

REVIEW_SYSTEM_PROMPT = """你要检查仓库答案是否使用了最小且完整的直接支持证据集合。
只返回一个 JSON 对象，不要使用 Markdown 代码围栏：
{"verdict":"pass","supported_citations":["E2"],"feedback":"..."}

规则：
- verdict 必须是 "pass" 或 "revise"。
- supported_citations 必须是提供的 evidence ID 的非空子集。
- 保留跨文件调用或 data-flow 解释所需的每个证据块。
- 删除不能直接支持答案的调用点、文档或相邻代码。
- 如果文档和直接实现代码重复表达同一事实，优先保留实现代码。
- 答案需要修改文字或引用时使用 "revise"，否则使用 "pass"。
- "revise" 只是有界 Answer Critic–Reviser 循环的反馈，不要求重新检索。
- feedback 必须是简洁的公开说明，不要包含隐藏推理。
"""

REVISION_SYSTEM_PROMPT = """你要在引用审查后修订源码仓库答案。
只能使用提供的证据，并只返回一个 JSON 对象，不要使用 Markdown 代码围栏：
{"outcome":"answered","answer":"...","citations":["E2"]}

规则：
- answer 必须简洁、非空，并使用与问题相同的语言。
- citations 只能包含提供的 evidence ID。
- 遵循审查反馈，并引用最小且完整的证据集合。
- 如果证据不足以支持答案，返回 outcome "insufficient_evidence"，简要解释无法支持的部分，
  并使用空的 citations 列表。
"""


def _evidence_blocks(results: Sequence[RankedChunk]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[E{index}] {chunk.relative_path}:{chunk.start_line}-{chunk.end_line}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_review_prompt(
    question: str,
    results: Sequence[RankedChunk],
    draft: ModelAnswer,
    *,
    evidence_groups: Sequence[tuple[str, Sequence[RankedChunk], Sequence[str]]] = (),
) -> tuple[str, str]:
    """根据草稿和检索到的证据构造引用审查 Prompt。"""
    evidence = (
        grouped_evidence_blocks(evidence_groups) if evidence_groups else _evidence_blocks(results)
    )
    user_prompt = (
        f"问题：\n{question}\n\n"
        f"答案草稿：\n{draft.answer}\n\n"
        f"草稿引用：\n{', '.join(draft.evidence_ids)}\n\n"
        f"仓库证据：\n{evidence}"
    )
    return REVIEW_SYSTEM_PROMPT, user_prompt


def build_revision_prompt(
    question: str,
    results: Sequence[RankedChunk],
    draft: ModelAnswer,
    review: CitationReview,
    *,
    evidence_groups: Sequence[tuple[str, Sequence[RankedChunk], Sequence[str]]] = (),
) -> tuple[str, str]:
    """根据公开审查反馈构造一次有界的答案修订 Prompt。"""
    evidence = (
        grouped_evidence_blocks(evidence_groups) if evidence_groups else _evidence_blocks(results)
    )
    user_prompt = (
        f"问题：\n{question}\n\n"
        f"原答案：\n{draft.answer}\n\n"
        f"原引用：\n{', '.join(draft.evidence_ids)}\n\n"
        f"审查反馈：\n{review.feedback}\n\n"
        f"直接支持的 evidence ID：\n{', '.join(review.supported_evidence_ids)}\n\n"
        f"仓库证据：\n{evidence}"
    )
    return REVISION_SYSTEM_PROMPT, user_prompt


def parse_citation_review(content: str, supplied_ids: frozenset[str]) -> CitationReview:
    """解析并校验一个结构化的引用审查结果。"""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelResponseError("引用审查结果不是有效 JSON") from error
    if not isinstance(payload, dict):
        raise ModelResponseError("引用审查结果必须是 JSON 对象")
    if frozenset(payload) != frozenset({"verdict", "supported_citations", "feedback"}):
        raise ModelResponseError("引用审查结果只能包含 verdict、supported_citations 和 feedback")

    verdict = payload.get("verdict")
    citations = payload.get("supported_citations")
    feedback = payload.get("feedback")
    if verdict not in SUPPORTED_REVIEW_VERDICTS:
        raise ModelResponseError("引用审查结果包含不支持的 verdict")
    if (
        not isinstance(citations, list)
        or not citations
        or not all(isinstance(item, str) for item in citations)
    ):
        raise ModelResponseError("引用审查结果必须包含支持的 evidence ID")
    unknown = [item for item in citations if item not in supplied_ids]
    if unknown:
        raise ModelResponseError(f"引用审查结果使用了未知 evidence ID：{unknown[0]}")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ModelResponseError("引用审查 feedback 不能为空")
    return CitationReview(
        verdict=verdict,
        supported_evidence_ids=tuple(dict.fromkeys(citations)),
        feedback=feedback.strip(),
    )
