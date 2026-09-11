# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「老路封闭」一节。
# =============================================================================
# """仓库回答 Agent 生成的不可变值对象。"""
#
# from dataclasses import dataclass
#
# from codeinsight.domain.answer import RepositoryAnswer, SubQuestionAnswer
#
# REVIEW_PASS = "pass"
# REVIEW_REVISE = "revise"
# SUPPORTED_REVIEW_VERDICTS = frozenset({REVIEW_PASS, REVIEW_REVISE})
#
#
# @dataclass(frozen=True)
# class CitationReview:
#     verdict: str
#     supported_evidence_ids: tuple[str, ...]
#     feedback: str
#
#
# @dataclass(frozen=True)
# class AgentEvent:
#     sequence: int
#     step: str
#     summary: str
#
#
# @dataclass(frozen=True)
# class AgentRepositoryAnswer:
#     result: RepositoryAnswer
#     revisions: int
#     events: tuple[AgentEvent, ...]
#     input_tokens: int
#     output_tokens: int
#     embedding_input_tokens: int = 0
#     subquestions: tuple[SubQuestionAnswer, ...] = ()
