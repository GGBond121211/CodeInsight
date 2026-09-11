# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「2026-09-11 老路整块注释冻结」一节。
# =============================================================================
# """有界引用修订 Agent 的应用层入口。"""
#
# from collections.abc import Callable, Sequence
# from pathlib import Path
#
# from codeinsight.agent.workflow import run_citation_agent
# from codeinsight.domain.agent import AgentRepositoryAnswer
# from codeinsight.domain.answer import ModelCompletion
# from codeinsight.domain.semantic import EmbeddingBatch
# from codeinsight.infrastructure.reranker import Reranker
#
# SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]
#
#
# def agent_answer_repository(
#     root: str | Path,
#     question: str,
#     *,
#     complete: Callable[[str, str], ModelCompletion],
#     limit: int = 5,
#     retrieval_mode: str = "hybrid",
#     semantic_embed: SemanticEmbed | None = None,
#     reranker: Reranker | None = None,
# ) -> AgentRepositoryAnswer:
#     """通过检索、引用审查和最多五次修订来回答问题。"""
#     return run_citation_agent(
#         root,
#         question,
#         complete=complete,
#         limit=limit,
#         retrieval_mode=retrieval_mode,
#         semantic_embed=semantic_embed,
#         reranker=reranker,
#     )
