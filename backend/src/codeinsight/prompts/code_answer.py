"""基于证据生成仓库回答的版本化 Prompt。"""

from collections.abc import Sequence

from codeinsight.domain.retrieval import RankedChunk

PROMPT_VERSION = "code-answer-v2"

EvidenceGroup = tuple[str, Sequence[RankedChunk], Sequence[str]]

SYSTEM_PROMPT = """你负责回答关于源码仓库的问题。
只能使用提供的证据，并使用与问题相同的语言回答。
只返回一个 JSON 对象，不要使用 Markdown 代码围栏：
{"outcome":"answered","answer":"...","citations":["E1"]}

规则：
- outcome 必须是 "answered" 或 "insufficient_evidence"。
- answer 字段必须始终包含非空的解释性句子。
- 当 outcome 为 "answered" 时，给出简洁解释，并引用回答所需的全部证据块。
- 只能引用直接支持答案的证据块。有代码证据时优先使用代码而不是文档；解释跨文件调用或
  data-flow 时，要包含支撑结论所需的全部实现代码块。
- citations 只能包含提供的 evidence ID，例如 E1 或 E2。
- 如果证据不足以支持请求的结论，使用 "insufficient_evidence"，简要说明无法支持的部分，
  不要猜测，并返回空的 citations 列表。
"""


def build_answer_prompt(
    question: str,
    results: Sequence[RankedChunk],
    evidence_ids: Sequence[str] | None = None,
) -> tuple[str, str]:
    """根据排序后的仓库证据构造稳定的 system/user Prompt。"""
    identifiers = tuple(evidence_ids or (f"E{index}" for index in range(1, len(results) + 1)))
    if len(identifiers) != len(results):
        raise ValueError("evidence ID 数量必须与结果数量一致")
    evidence = _evidence_blocks(results, identifiers)
    user_prompt = f"问题：\n{question}\n\n仓库证据：\n{evidence}"
    return SYSTEM_PROMPT, user_prompt


def _evidence_blocks(
    results: Sequence[RankedChunk],
    evidence_ids: Sequence[str] | None = None,
) -> str:
    identifiers = tuple(evidence_ids or (f"E{index}" for index in range(1, len(results) + 1)))
    if len(identifiers) != len(results):
        raise ValueError("evidence ID 数量必须与结果数量一致")
    blocks = []
    for evidence_id, result in zip(identifiers, results, strict=True):
        chunk = result.chunk
        blocks.append(
            f"[{evidence_id}] {chunk.relative_path}:{chunk.start_line}-{chunk.end_line}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def grouped_evidence_blocks(groups: Sequence[EvidenceGroup]) -> str:
    """在保留每个 subquestion 局部边界的前提下格式化证据。"""
    sections = []
    for question, results, evidence_ids in groups:
        sections.append(f"子问题：\n{question}\n\n{_evidence_blocks(results, evidence_ids)}")
    return "\n\n".join(sections)


def build_grouped_answer_prompt(
    question: str,
    groups: Sequence[EvidenceGroup],
) -> tuple[str, str]:
    """构造按 subquestion 分开维护证据覆盖的 Prompt。"""
    evidence = grouped_evidence_blocks(groups)
    user_prompt = (
        f"问题：\n{question}\n\n"
        "请分别回答每个子问题。仓库证据按子问题分组；不要悄悄使用一个分组的证据回答另一个分组的问题：\n"
        f"{evidence}"
    )
    return SYSTEM_PROMPT, user_prompt
