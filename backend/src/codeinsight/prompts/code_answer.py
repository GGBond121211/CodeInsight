"""Versioned prompt for evidence-grounded repository answers."""

from collections.abc import Sequence

from codeinsight.domain.retrieval import RankedChunk

PROMPT_VERSION = "code-answer-v2"

EvidenceGroup = tuple[str, Sequence[RankedChunk], Sequence[str]]

SYSTEM_PROMPT = """You answer questions about a source repository.
Use only the supplied evidence. Answer in the same language as the question.
Return exactly one JSON object and no Markdown fence:
{"outcome":"answered","answer":"...","citations":["E1"]}

Rules:
- outcome must be "answered" or "insufficient_evidence".
- The answer field must always contain a non-empty explanatory sentence.
- For "answered", give a concise explanation and cite every evidence block needed.
- Cite only blocks that directly support the answer. Prefer code over documentation
  when code evidence is available, and include every implementation block needed
  to support a cross-file call or data-flow explanation.
- Citations must contain only supplied evidence IDs such as E1 or E2.
- If the evidence cannot support the requested conclusion, use
  "insufficient_evidence", briefly state what is unsupported without guessing,
  and use an empty citations list.
"""


def build_answer_prompt(
    question: str,
    results: Sequence[RankedChunk],
    evidence_ids: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Build stable system and user prompts from ranked repository evidence."""
    identifiers = tuple(evidence_ids or (f"E{index}" for index in range(1, len(results) + 1)))
    if len(identifiers) != len(results):
        raise ValueError("evidence ID count must match result count")
    evidence = _evidence_blocks(results, identifiers)
    user_prompt = f"Question:\n{question}\n\nRepository evidence:\n{evidence}"
    return SYSTEM_PROMPT, user_prompt


def _evidence_blocks(
    results: Sequence[RankedChunk],
    evidence_ids: Sequence[str] | None = None,
) -> str:
    identifiers = tuple(evidence_ids or (f"E{index}" for index in range(1, len(results) + 1)))
    if len(identifiers) != len(results):
        raise ValueError("evidence ID count must match result count")
    blocks = []
    for evidence_id, result in zip(identifiers, results, strict=True):
        chunk = result.chunk
        blocks.append(
            f"[{evidence_id}] {chunk.relative_path}:{chunk.start_line}-{chunk.end_line}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def grouped_evidence_blocks(groups: Sequence[EvidenceGroup]) -> str:
    """Format evidence while preserving each subquestion's local boundary."""
    sections = []
    for question, results, evidence_ids in groups:
        sections.append(f"Subquestion:\n{question}\n\n{_evidence_blocks(results, evidence_ids)}")
    return "\n\n".join(sections)


def build_grouped_answer_prompt(
    question: str,
    groups: Sequence[EvidenceGroup],
) -> tuple[str, str]:
    """Build a prompt that keeps evidence coverage separate per subquestion."""
    evidence = grouped_evidence_blocks(groups)
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Answer each subquestion separately. Repository evidence is grouped by subquestion; "
        "do not use evidence from one group to silently answer another:\n"
        f"{evidence}"
    )
    return SYSTEM_PROMPT, user_prompt
