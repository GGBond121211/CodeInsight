"""Evaluation-only prompts for structured Temporary-30 answer ablations."""

from __future__ import annotations

import json
from collections.abc import Sequence

from codeinsight.domain.retrieval import RankedChunk

STRUCTURED_PROMPT_VERSION = "temporary-30-structured-v1"
STRUCTURED_REVIEW_VERSION = "temporary-30-critic-v1"

DRAFT_SYSTEM_PROMPT = """You answer questions about a source repository using only
supplied evidence.
Return exactly one JSON object and no Markdown fence. The required shape is:
{"case_id":"...","subquestion_outputs":[{"subquestion_id":"Q1","outcome":"answered","answer":"...","citations":["Q1E1"]}]}

Rules:
- Include every supplied subquestion exactly once and preserve its subquestion_id.
- outcome must be "answered" or "insufficient_evidence".
- Answer each subquestion independently and in the language used by that subquestion.
- Cite only evidence IDs from that subquestion's evidence group.
- For answered, cite every block needed to support the conclusion and cross-file flow.
- For insufficient_evidence, explain what the repository cannot prove and use [].
- Prefer implementation code over conflicting documentation while explicitly describing conflicts.
- Do not infer frameworks, external calls, persistence, tax, discounts, or state
  transitions not shown.
"""

REVIEW_SYSTEM_PROMPT = """You are the evidence critic for structured repository answers.
Return exactly one JSON object and no Markdown fence. The required shape is:
{"subquestion_reviews":[{"subquestion_id":"Q1","verdict":"pass","supported_citations":["Q1E1"],"feedback":"..."}]}

Rules:
- Review every supplied subquestion exactly once and preserve its subquestion_id.
- verdict must be "pass" or "revise".
- Check the conclusion as well as citation validity, completeness, and evidence-group ownership.
- Use revise for a false conclusion, unsupported assertion, missing necessary
  evidence, or cross-group citation.
- supported_citations must only contain evidence IDs owned by that subquestion.
- An insufficient_evidence answer may pass with an empty supported_citations list.
- feedback must be concise public correction instructions, never hidden reasoning.
"""

REVISION_SYSTEM_PROMPT = """You revise a structured repository answer from critic feedback.
Return exactly one JSON object with the same shape and case_id as the draft:
{"case_id":"...","subquestion_outputs":[{"subquestion_id":"Q1","outcome":"answered","answer":"...","citations":["Q1E1"]}]}

Rules:
- Preserve every subquestion_id exactly once and keep their original order.
- Change only subquestions whose critic verdict is revise; copy pass entries without alteration.
- Use only evidence from each subquestion's own group.
- Follow feedback, correct the conclusion, and cite the smallest complete evidence set.
- Use insufficient_evidence with [] when the evidence cannot support the requested conclusion.
- Return JSON only, with no commentary or Markdown fence.
"""


def _evidence_payload(
    subquestions: Sequence[dict],
    evidence_groups: Sequence[tuple[Sequence[RankedChunk], Sequence[str]]],
) -> list[dict]:
    payload: list[dict] = []
    for subquestion, (results, evidence_ids) in zip(subquestions, evidence_groups, strict=True):
        if len(results) != len(evidence_ids):
            raise ValueError("evidence IDs must match ranked results")
        payload.append(
            {
                "subquestion_id": subquestion["id"],
                "question": subquestion["question"],
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "path": result.chunk.relative_path,
                        "start_line": result.chunk.start_line,
                        "end_line": result.chunk.end_line,
                        "text": result.chunk.text,
                    }
                    for result, evidence_id in zip(results, evidence_ids, strict=True)
                ],
            }
        )
    return payload


def build_structured_draft_prompt(
    case_id: str,
    original_question: str,
    subquestions: Sequence[dict],
    evidence_groups: Sequence[tuple[Sequence[RankedChunk], Sequence[str]]],
) -> tuple[str, str]:
    user_payload = {
        "case_id": case_id,
        "original_question": original_question,
        "subquestions": _evidence_payload(subquestions, evidence_groups),
    }
    return DRAFT_SYSTEM_PROMPT, json.dumps(user_payload, ensure_ascii=False)


def build_structured_review_prompt(
    case_id: str,
    original_question: str,
    subquestions: Sequence[dict],
    evidence_groups: Sequence[tuple[Sequence[RankedChunk], Sequence[str]]],
    draft_payload: dict,
) -> tuple[str, str]:
    user_payload = {
        "case_id": case_id,
        "original_question": original_question,
        "subquestions": _evidence_payload(subquestions, evidence_groups),
        "draft": draft_payload,
    }
    return REVIEW_SYSTEM_PROMPT, json.dumps(user_payload, ensure_ascii=False)


def build_structured_revision_prompt(
    case_id: str,
    original_question: str,
    subquestions: Sequence[dict],
    evidence_groups: Sequence[tuple[Sequence[RankedChunk], Sequence[str]]],
    draft_payload: dict,
    review_payload: dict,
) -> tuple[str, str]:
    user_payload = {
        "case_id": case_id,
        "original_question": original_question,
        "subquestions": _evidence_payload(subquestions, evidence_groups),
        "current_answer": draft_payload,
        "critic_review": review_payload,
    }
    return REVISION_SYSTEM_PROMPT, json.dumps(user_payload, ensure_ascii=False)
