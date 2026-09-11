# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「2026-09-11 老路整块注释冻结」一节。
# =============================================================================
# from dataclasses import FrozenInstanceError
#
# import pytest
#
# from codeinsight.domain.agent import AgentEvent, AgentRepositoryAnswer, CitationReview
# from codeinsight.domain.answer import RepositoryAnswer
#
#
# def test_agent_values_are_immutable() -> None:
#     answer = RepositoryAnswer(
#         outcome="insufficient_evidence",
#         answer="Not enough evidence.",
#         citations=(),
#         retrieval_mode="hybrid",
#         model="test-model",
#         prompt_version="citation-agent-v3",
#         input_tokens=10,
#         output_tokens=5,
#     )
#     result = AgentRepositoryAnswer(
#         result=answer,
#         revisions=0,
#         events=(AgentEvent(1, "retrieve", "Found no evidence."),),
#         input_tokens=10,
#         output_tokens=5,
#     )
#
#     with pytest.raises(FrozenInstanceError):
#         result.revisions = 1  # type: ignore[misc]
#
#
# def test_review_holds_supported_evidence_and_public_feedback() -> None:
#     review = CitationReview("revise", ("E2",), "Remove the unrelated call site.")
#
#     assert review.supported_evidence_ids == ("E2",)
#     assert review.feedback.startswith("Remove")
