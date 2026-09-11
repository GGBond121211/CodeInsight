# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「老路封闭」一节。
# =============================================================================
# import pytest
#
# from codeinsight.domain.answer import ModelAnswer
# from codeinsight.domain.errors import ModelResponseError
# from codeinsight.domain.retrieval import RankedChunk
# from codeinsight.domain.source import SourceChunk
# from codeinsight.prompts.citation_review import (
#     AGENT_PROMPT_VERSION,
#     CITATION_REVIEW_VERSION,
#     build_review_prompt,
#     build_revision_prompt,
#     parse_citation_review,
# )
#
#
# def _results() -> tuple[RankedChunk, ...]:
#     return (
#         RankedChunk(SourceChunk("src/api.py", 1, 4, "checkout(request)"), 2.0, 1),
#         RankedChunk(SourceChunk("src/service.py", 5, 9, "def checkout(): pass"), 1.5, 2),
#     )
#
#
# def _draft() -> ModelAnswer:
#     return ModelAnswer("answered", "checkout is defined in service.py.", ("E1", "E2"), "m", 1, 1)
#
#
# def test_review_and_revision_prompts_preserve_evidence_ids() -> None:
#     review_system, review_user = build_review_prompt("Where?", _results(), _draft())
#     review = parse_citation_review(
#         '{"verdict":"revise","supported_citations":["E2"],"feedback":"Remove E1."}',
#         frozenset({"E1", "E2"}),
#     )
#     revision_system, revision_user = build_revision_prompt("Where?", _results(), _draft(), review)
#
#     assert CITATION_REVIEW_VERSION == "citation-review-v2"
#     assert AGENT_PROMPT_VERSION == "citation-agent-v3"
#     assert "[E2] src/service.py:5-9" in review_user
#     assert "最小且完整" in review_system
#     assert "Remove E1." in revision_user
#     assert '"citations":["E2"]' in revision_system
#
#
# @pytest.mark.parametrize(
#     "content",
#     [
#         "not json",
#         "[]",
#         '{"verdict":"unknown","supported_citations":["E1"],"feedback":"x"}',
#         '{"verdict":"pass","supported_citations":[],"feedback":"x"}',
#         '{"verdict":"pass","supported_citations":["E99"],"feedback":"x"}',
#         '{"verdict":"pass","supported_citations":["E1"],"feedback":""}',
#     ],
# )
# def test_invalid_review_raises(content: str) -> None:
#     with pytest.raises(ModelResponseError):
#         parse_citation_review(content, frozenset({"E1", "E2"}))
