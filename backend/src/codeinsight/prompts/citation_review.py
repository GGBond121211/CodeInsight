"""Versioned prompts and parser for citation review and answer revision."""

import json
from collections.abc import Sequence

from codeinsight.domain.agent import SUPPORTED_REVIEW_VERDICTS, CitationReview
from codeinsight.domain.answer import ModelAnswer
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.prompts.code_answer import grouped_evidence_blocks

CITATION_REVIEW_VERSION = "citation-review-v2"
AGENT_PROMPT_VERSION = "citation-agent-v3"

REVIEW_SYSTEM_PROMPT = """You verify whether a repository answer uses the smallest complete set
of directly supporting evidence. Return exactly one JSON object and no Markdown fence:
{"verdict":"pass","supported_citations":["E2"],"feedback":"..."}

Rules:
- verdict must be "pass" or "revise".
- supported_citations must be a non-empty subset of supplied evidence IDs.
- Keep every block needed for a cross-file call or data-flow explanation.
- Remove call sites, documentation, or neighboring code that does not directly support the answer.
- Prefer direct implementation code when documentation repeats the same fact.
- Use "revise" when the answer needs wording or citation changes; otherwise use "pass".
- A "revise" verdict is feedback for the bounded Answer Critic–Reviser loop;
  it does not request a new retrieval attempt.
- feedback must be a concise public explanation. Do not include hidden reasoning.
"""

REVISION_SYSTEM_PROMPT = """You revise an answer about a source repository after citation review.
Use only supplied evidence and return exactly one JSON object with no Markdown fence:
{"outcome":"answered","answer":"...","citations":["E2"]}

Rules:
- The answer must be concise, non-empty, and in the same language as the question.
- citations must contain only supplied evidence IDs.
- Follow the review feedback and cite the smallest complete evidence set.
- If the evidence cannot support an answer, return outcome "insufficient_evidence",
  briefly explain what is unsupported, and use an empty citations list.
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
    """Build a citation review prompt from the draft and retrieved evidence."""
    evidence = (
        grouped_evidence_blocks(evidence_groups) if evidence_groups else _evidence_blocks(results)
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Draft answer:\n{draft.answer}\n\n"
        f"Draft citations:\n{', '.join(draft.evidence_ids)}\n\n"
        f"Repository evidence:\n{evidence}"
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
    """Build one bounded answer-revision prompt from public review feedback."""
    evidence = (
        grouped_evidence_blocks(evidence_groups) if evidence_groups else _evidence_blocks(results)
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Original answer:\n{draft.answer}\n\n"
        f"Original citations:\n{', '.join(draft.evidence_ids)}\n\n"
        f"Review feedback:\n{review.feedback}\n\n"
        f"Directly supported evidence IDs:\n{', '.join(review.supported_evidence_ids)}\n\n"
        f"Repository evidence:\n{evidence}"
    )
    return REVISION_SYSTEM_PROMPT, user_prompt


def parse_citation_review(content: str, supplied_ids: frozenset[str]) -> CitationReview:
    """Parse and validate one structured citation review."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelResponseError("citation review is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ModelResponseError("citation review must be a JSON object")
    if frozenset(payload) != frozenset({"verdict", "supported_citations", "feedback"}):
        raise ModelResponseError(
            "citation review must contain only verdict, supported_citations, and feedback"
        )

    verdict = payload.get("verdict")
    citations = payload.get("supported_citations")
    feedback = payload.get("feedback")
    if verdict not in SUPPORTED_REVIEW_VERDICTS:
        raise ModelResponseError("citation review has an unsupported verdict")
    if (
        not isinstance(citations, list)
        or not citations
        or not all(isinstance(item, str) for item in citations)
    ):
        raise ModelResponseError("citation review must include supported evidence IDs")
    unknown = [item for item in citations if item not in supplied_ids]
    if unknown:
        raise ModelResponseError(f"citation review used unknown evidence ID: {unknown[0]}")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ModelResponseError("citation review feedback must be non-empty")
    return CitationReview(
        verdict=verdict,
        supported_evidence_ids=tuple(dict.fromkeys(citations)),
        feedback=feedback.strip(),
    )
