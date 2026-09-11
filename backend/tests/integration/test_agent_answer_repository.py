# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「2026-09-11 老路整块注释冻结」一节。
# =============================================================================
# from pathlib import Path
#
# import pytest
#
# from codeinsight.application.agent_answer_repository import agent_answer_repository
# from codeinsight.domain.answer import ModelCompletion
# from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
# from codeinsight.infrastructure.reranker import RerankResult
#
# # 2026-09-10：agent_answer_repository 是冻结路线（LangGraph）的适配层，
# # 默认不运行本文件；复核历史实现时用 pytest -m legacy。
# pytestmark = pytest.mark.legacy
#
# BACKEND_ROOT = Path(__file__).resolve().parents[2]
# FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
#
#
# def _fake_embed(texts):
#     vectors = []
#     for _ in texts:
#         vectors.append((1.0, 0.0))
#     return EmbeddingBatch(
#         "fake",
#         tuple(vectors),
#         len(texts),
#         tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
#         "dense-sparse-v1",
#     )
#
#
# class CompletionSequence:
#     def __init__(self, *contents: str) -> None:
#         self.contents = list(contents)
#         self.index = 0
#
#     def __call__(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
#         content = self.contents[self.index]
#         self.index += 1
#         return ModelCompletion(content, "fake-model", 12, 5)
#
#
# class _FakeReranker:
#     def rerank(self, _query, documents, *, top_n):
#         return tuple(RerankResult(index, 1.0) for index in range(min(top_n, len(documents))))
#
#
# def test_agent_answer_maps_reviewed_evidence_from_real_retrieval() -> None:
#     complete = CompletionSequence(
#         '{"outcome":"answered","answer":"checkout validates input.","citations":["E1","E2"]}',
#         '{"verdict":"pass","supported_citations":["E1","E2"],"feedback":"Both are direct."}',
#     )
#
#     result = agent_answer_repository(
#         FIXTURE_ROOT,
#         "How does checkout validate input?",
#         complete=complete,
#         semantic_embed=_fake_embed,
#         reranker=_FakeReranker(),
#     )
#
#     assert result.result.outcome == "answered"
#     assert result.result.retrieval_mode == "hybrid"
#     assert result.result.prompt_version == "citation-agent-v3"
#     assert result.result.citations
#     for citation in result.result.citations:
#         assert citation.relative_path.startswith("src/shop/")
#     assert result.revisions == 0
