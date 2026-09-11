# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「2026-09-11 老路整块注释冻结」一节。
# =============================================================================
# """QueryPlan 感知的 LangGraph 检索状态测试。"""
#
# from pathlib import Path
#
# import pytest
#
# from codeinsight.agent.workflow import run_citation_agent
# from codeinsight.domain.answer import ModelCompletion
# from codeinsight.domain.query_plan import QueryPlan, SubQuestion
# from codeinsight.domain.retrieval import RankedChunk
# from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
# from codeinsight.domain.source import SourceChunk
# from codeinsight.infrastructure.reranker import RerankResult
#
# # 2026-09-10：explain 已统一走只读 Tool Loop，本文件覆盖的是冻结的 LangGraph
# # 实现。默认不运行（见 backend/pyproject.toml 的 addopts），复核历史实现时用
# # pytest -m legacy。
# pytestmark = pytest.mark.legacy
#
# BACKEND_ROOT = Path(__file__).resolve().parents[3]
# FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
#
#
# class _CompletionSequence:
#     def __init__(self, *contents: str) -> None:
#         self.contents = list(contents)
#         self.calls = 0
#         self.prompts: list[str] = []
#
#     def __call__(self, _system: str, user: str) -> ModelCompletion:
#         self.prompts.append(user)
#         content = self.contents[self.calls]
#         self.calls += 1
#         return ModelCompletion(content, "fake-agent", 10, 4)
#
#
# def _fake_embedding(texts):
#     vectors: list[tuple[float, float]] = []
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
# class _FakeReranker:
#     def rerank(self, _query, documents, *, top_n):
#         return tuple(RerankResult(index, 1.0) for index in range(min(top_n, len(documents))))
#
#
# def test_query_plan_agent_retrieves_each_subquestion_and_deduplicates_state() -> None:
#     plan = QueryPlan(
#         original_question="checkout 怎么校验，价格怎么算？",
#         language="mixed",
#         normalized_question="Explain validation and pricing.",
#         subquestions=(
#             SubQuestion("How does checkout validate input?", "implementation", "hybrid"),
#             SubQuestion("How does checkout compute the total?", "data_flow", "hybrid"),
#         ),
#         retrieval_modes=("hybrid",),
#         execution_route="agent",
#         confidence=0.9,
#     )
#     calls: list[tuple[str, str]] = []
#
#     def search(_root, question, *, retrieval_mode, **_kwargs):
#         calls.append((question, retrieval_mode))
#         return (
#             RankedChunk(
#                 SourceChunk("src/shop/service.py", 10, 19, "checkout evidence"),
#                 1.0,
#                 1,
#             ),
#         )
#
#     complete = _CompletionSequence(
#         '{"outcome":"answered","answer":"Validation is in checkout.","citations":["E1"]}',
#         '{"outcome":"insufficient_evidence","answer":"The total formula is not shown.",'
#         '"citations":[]}',
#         '{"verdict":"pass","supported_citations":["E1"],"feedback":"Supported."}',
#     )
#
#     result = run_citation_agent(
#         FIXTURE_ROOT,
#         plan.original_question,
#         complete=complete,
#         retrieval_mode="auto",
#         search=search,
#         query_plan=plan,
#         semantic_embed=_fake_embedding,
#         reranker=_FakeReranker(),
#     )
#
#     assert calls == [
#         ("How does checkout validate input?", "hybrid"),
#         ("How does checkout compute the total?", "hybrid"),
#     ]
#     assert result.result.retrieval_mode == "auto"
#     assert result.result.outcome == "partially_answered"
#     assert result.result.citations[0].evidence_id == "E1"
#     outcomes = []
#     for item in result.subquestions:
#         outcomes.append(item.outcome)
#     assert outcomes == [
#         "answered",
#         "insufficient_evidence",
#     ]
#     assert result.subquestions[0].answer == "Validation is in checkout."
#     assert result.subquestions[1].answer == "The total formula is not shown."
#     assert result.subquestions[1].citations == ()
#     assert "How does checkout compute the total?" not in complete.prompts[0]
#     assert "How does checkout validate input?" not in complete.prompts[1]
#     assert "子问题" in result.events[0].summary
#     assert complete.calls == 3
#
#
# def test_query_plan_critic_revises_only_failed_subquestion() -> None:
#     plan = QueryPlan(
#         original_question="Explain validation and pricing.",
#         language="en",
#         normalized_question="Explain validation and pricing.",
#         subquestions=(
#             SubQuestion("How is input validated?", "implementation", "hybrid"),
#             SubQuestion("How is price computed?", "data_flow", "hybrid"),
#         ),
#         retrieval_modes=("hybrid",),
#         execution_route="agent",
#         confidence=0.9,
#     )
#
#     def search(_root, question, *, retrieval_mode, **_kwargs):
#         line = 1 if "validated" in question else 10
#         path = "src/validation.py" if "validated" in question else "src/pricing.py"
#         return (
#             RankedChunk(
#                 SourceChunk(path, line, line + 2, f"evidence for {question}"),
#                 1.0,
#                 1,
#             ),
#         )
#
#     complete = _CompletionSequence(
#         '{"outcome":"answered","answer":"Validation answer.","citations":["E1"]}',
#         '{"outcome":"answered","answer":"Weak price answer.","citations":["E2"]}',
#         '{"verdict":"pass","supported_citations":["E1"],"feedback":"Supported."}',
#         '{"verdict":"revise","supported_citations":["E2"],"feedback":"Be precise."}',
#         '{"outcome":"answered","answer":"Precise price answer.","citations":["E2"]}',
#         '{"verdict":"pass","supported_citations":["E2"],"feedback":"Supported."}',
#     )
#
#     result = run_citation_agent(
#         FIXTURE_ROOT,
#         plan.original_question,
#         complete=complete,
#         retrieval_mode="auto",
#         search=search,
#         query_plan=plan,
#         semantic_embed=_fake_embedding,
#         reranker=_FakeReranker(),
#     )
#
#     assert result.revisions == 1
#     answers = []
#     for item in result.subquestions:
#         answers.append(item.answer)
#     assert answers == [
#         "Validation answer.",
#         "Precise price answer.",
#     ]
#     validation_prompt_count = 0
#     pricing_prompt_count = 0
#     for prompt in complete.prompts:
#         if "How is input validated?" in prompt:
#             validation_prompt_count += 1
#         if "How is price computed?" in prompt:
#             pricing_prompt_count += 1
#     assert validation_prompt_count == 2
#     assert pricing_prompt_count == 4
#     assert complete.calls == 6
