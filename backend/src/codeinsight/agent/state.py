# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「2026-09-11 老路整块注释冻结」一节。
# =============================================================================
# """引用修订 Agent 使用的内部 LangGraph 状态。"""
#
# from pathlib import Path
# from typing import TypedDict
#
# from codeinsight.domain.agent import AgentEvent, AgentRepositoryAnswer, CitationReview
# from codeinsight.domain.answer import ModelAnswer
# from codeinsight.domain.query_plan import QueryPlan
# from codeinsight.domain.retrieval import RankedChunk, SubQuestionEvidence
#
#
# class CitationAgentState(TypedDict, total=False):
#     repository_root: str | Path
#     question: str
#     limit: int
#     retrieval_mode: str
#     query_plan: QueryPlan
#     route_reason: str
#     router_confidence: float
#     subquestion_results: tuple[tuple[str, tuple[RankedChunk, ...]], ...]
#     subquestion_evidence: tuple[SubQuestionEvidence, ...]
#     results: tuple[RankedChunk, ...]
#     draft: ModelAnswer
#     subquestion_drafts: tuple[ModelAnswer, ...]
#     review: CitationReview
#     subquestion_reviews: tuple[CitationReview | None, ...]
#     subquestion_accepted: tuple[bool, ...]
#     revised: ModelAnswer
#     revisions: int
#     max_revisions: int
#     events: tuple[AgentEvent, ...]
#     input_tokens: int
#     output_tokens: int
#     agent_result: AgentRepositoryAnswer
