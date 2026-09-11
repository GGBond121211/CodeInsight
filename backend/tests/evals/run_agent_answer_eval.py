# =============================================================================
# 【已冻结 · 仅作历史实现保留】旧 LangGraph 代码理解路线
# 2026-09-10 起 explain 不再调用它；2026-09-11 起按用户要求整块注释掉，
# 但**不删除**：正常运行时和测试都不再执行这里的任何一行。
# 需要复核或回退时，取消本文件的行首 "# " 即可；改动说明见
# docs/RELEASE_2_1_0_Q8_Q9.md 的「2026-09-11 老路整块注释冻结」一节。
# =============================================================================
# """Run the live LangGraph citation Agent evaluation."""
#
# import json
# import os
# import time
# from dataclasses import asdict
# from pathlib import Path
# from urllib.parse import urlsplit
#
# from codeinsight.application.agent_answer_repository import agent_answer_repository
# from codeinsight.domain.agent import AgentRepositoryAnswer
# from codeinsight.domain.answer import RepositoryAnswer
# from codeinsight.domain.errors import ModelCallError, ModelResponseError
# from codeinsight.evaluation.answer_metrics import (
#     answer_failure_types,
#     evaluate_answer_results,
# )
# from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
# from codeinsight.infrastructure.openai_chat import OpenAIChatModel
# from codeinsight.prompts.citation_review import AGENT_PROMPT_VERSION
#
# BACKEND_ROOT = Path(__file__).resolve().parents[2]
# SOURCE_CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "cases.json"
# ANSWER_CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "answer_cases.json"
# FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
# OUTPUT_PATH = BACKEND_ROOT.parent / "outputs" / "evals" / "agent-answer-baseline.json"
#
#
# def load_agent_cases() -> list[dict]:
#     source_document = json.loads(SOURCE_CASES_PATH.read_text(encoding="utf-8"))
#     answer_document = json.loads(ANSWER_CASES_PATH.read_text(encoding="utf-8"))
#     source_by_id = {}
#     for case in source_document["cases"]:
#         source_by_id[case["id"]] = case
#     resolved = []
#     for selected in answer_document["cases"]:
#         case = dict(source_by_id[selected["id"]])
#         case["required_terms"] = list(selected["required_terms"])
#         resolved.append(case)
#     return resolved
#
#
# def build_agent_evaluation_payload(
#     *,
#     cases: list[dict],
#     results: dict[str, AgentRepositoryAnswer | None],
#     errors: dict[str, str],
#     elapsed_milliseconds: dict[str, int],
#     fixture_root: Path,
#     model: str,
#     base_url_host: str | None,
# ) -> dict:
#     repository_results: dict[str, RepositoryAnswer | None] = {}
#     for case_id, result in results.items():
#         repository_results[case_id] = result.result if result else None
#     metrics = evaluate_answer_results(cases, repository_results, fixture_root)
#     case_payloads = []
#     total_input_tokens = 0
#     total_output_tokens = 0
#     total_revisions = 0
#     for case in cases:
#         case_id = case["id"]
#         agent_result = results.get(case_id)
#         result = agent_result.result if agent_result else None
#         if agent_result:
#             total_input_tokens += agent_result.input_tokens
#             total_output_tokens += agent_result.output_tokens
#             total_revisions += agent_result.revisions
#         citations = []
#         events = []
#         if result:
#             for citation in result.citations:
#                 citations.append(asdict(citation))
#         if agent_result:
#             for event in agent_result.events:
#                 events.append(asdict(event))
#         case_payloads.append(
#             {
#                 "case_id": case_id,
#                 "question": case["input"]["question"],
#                 "expected_outcome": case["expected"]["outcome"],
#                 "outcome": result.outcome if result else None,
#                 "answer": result.answer if result else None,
#                 "citations": citations,
#                 "revisions": agent_result.revisions if agent_result else 0,
#                 "events": events,
#                 "input_tokens": agent_result.input_tokens if agent_result else None,
#                 "output_tokens": agent_result.output_tokens if agent_result else None,
#                 "elapsed_milliseconds": elapsed_milliseconds[case_id],
#                 "failure_types": answer_failure_types(case, result, fixture_root),
#                 "error": errors.get(case_id),
#             }
#         )
#
#     payload = {
#         "agent_version": AGENT_PROMPT_VERSION,
#         "linear_baseline": "code-answer-v2",
#         "retrieval_mode": "hybrid",
#         "model": model,
#         "case_count": len(cases),
#         "metrics": asdict(metrics),
#         "usage": {
#             "input_tokens": total_input_tokens,
#             "output_tokens": total_output_tokens,
#         },
#         "total_revisions": total_revisions,
#         "cases": case_payloads,
#     }
#     if base_url_host:
#         payload["base_url_host"] = base_url_host
#     return payload
#
#
# def main() -> int:
#     cases = load_agent_cases()
#     model = OpenAIChatModel.from_environment()
#     embedding_model = OpenAIEmbeddingModel.from_environment()
#     results: dict[str, AgentRepositoryAnswer | None] = {}
#     errors: dict[str, str] = {}
#     elapsed_milliseconds: dict[str, int] = {}
#
#     for case in cases:
#         started = time.perf_counter()
#         try:
#             results[case["id"]] = agent_answer_repository(
#                 FIXTURE_ROOT,
#                 case["input"]["question"],
#                 complete=model.complete,
#                 retrieval_mode="hybrid",
#                 semantic_embed=embedding_model.embed,
#                 limit=5,
#             )
#         except (ModelCallError, ModelResponseError, OSError, ValueError) as error:
#             results[case["id"]] = None
#             errors[case["id"]] = str(error)
#         elapsed_milliseconds[case["id"]] = round((time.perf_counter() - started) * 1000)
#
#     configured_url = os.environ.get("CODEINSIGHT_BASE_URL", "")
#     base_url_host = urlsplit(configured_url).hostname if configured_url else None
#     payload = build_agent_evaluation_payload(
#         cases=cases,
#         results=results,
#         errors=errors,
#         elapsed_milliseconds=elapsed_milliseconds,
#         fixture_root=FIXTURE_ROOT,
#         model=model.model,
#         base_url_host=base_url_host,
#     )
#     OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
#     OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
#     summary = {
#         "agent_version": payload["agent_version"],
#         "linear_baseline": payload["linear_baseline"],
#         "model": payload["model"],
#         "case_count": payload["case_count"],
#         "metrics": payload["metrics"],
#         "usage": payload["usage"],
#         "total_revisions": payload["total_revisions"],
#         "failed_case_ids": failed_case_ids(payload["cases"]),
#     }
#     print(json.dumps(summary, indent=2))
#     return 0
#
#
# def failed_case_ids(case_payloads: list[dict]) -> list[str]:
#     failed = []
#     for item in case_payloads:
#         if item["failure_types"]:
#             failed.append(item["case_id"])
#     return failed
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())
