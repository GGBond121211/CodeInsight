"""CodeInsight 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.context_assembler import ContextAssembler
from codeinsight.application.query_router import route_question
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.model_gateway import GatewayChatModel as OpenAIChatModel
from codeinsight.infrastructure.reranker import OpenAITextReranker

PROJECT_DESCRIPTION = "CodeInsight 理解源码仓库，并提供可核验的文件和行号证据。"


def _configured_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel.from_environment(context_assembler=ContextAssembler())


def _positive_int(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit 必须是正整数")
    return limit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeinsight", description=PROJECT_DESCRIPTION)
    commands = parser.add_subparsers(dest="command")
    auto_answer = commands.add_parser("auto-answer", help="路由多语言问题，并返回公开计划元数据")
    auto_answer.add_argument("--repo", required=True, metavar="PATH")
    auto_answer.add_argument("question", metavar="QUESTION")
    auto_answer.add_argument("--limit", type=_positive_int, default=5, metavar="N")
    auto_answer.add_argument(
        "--force-route",
        choices=("linear", "agent"),
        default=None,
        dest="force_route",
        help="覆盖 Router 选择的公开执行路线",
    )
    demo = commands.add_parser("change-demo", help="无需模型的审批、Sandbox、回滚契约演示")
    demo.add_argument("--work-dir", required=True, metavar="PATH")
    return parser


def _auto_answer(repo: str, question: str, limit: int, force_route: str | None) -> int:
    try:
        model = _configured_chat_model()
        router_result = route_question(question, complete=model.complete)
        if force_route:
            plan = replace(
                router_result.plan,
                execution_route=force_route,
                fallback_reason="forced_route",
            )
            router_result = replace(
                router_result,
                plan=plan,
                used_fallback=False,
                fallback_reason="forced_route",
            )
        embedding_model = (
            OpenAIEmbeddingModel.from_environment()
            if router_result.plan.execution_route != "insufficient"
            else None
        )
        reranker = (
            OpenAITextReranker.from_environment()
            if router_result.plan.execution_route != "insufficient"
            else None
        )
        if router_result.plan.execution_route == "agent":
            agent_result = run_citation_agent(
                repo,
                question,
                complete=model.complete,
                limit=limit,
                retrieval_mode="auto",
                semantic_embed=embedding_model.embed if embedding_model else None,
                reranker=reranker,
                query_plan=router_result.plan,
            )
            result = agent_result.result
            events = agent_result.events
        else:
            auto_result = auto_answer_repository(
                repo,
                router_result=router_result,
                generate=model.generate,
                limit=limit,
                semantic_embed=embedding_model.embed if embedding_model else None,
                reranker=reranker,
            )
            result = auto_result
            events = auto_result.events
    except ModelConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (ModelCallError, ModelResponseError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(result.answer)
    if result.citations:
        print("证据：")
        for citation in result.citations:
            print(
                f"- [{citation.evidence_id}] {citation.relative_path}:"
                f"{citation.start_line}-{citation.end_line}"
            )
    print(f"执行路线：{router_result.plan.execution_route}")
    print(
        f"Router：模型={router_result.model or '未调用'} "
        f"Token={router_result.input_tokens}/{router_result.output_tokens} "
        f"耗时_ms={router_result.elapsed_milliseconds:.1f}"
    )
    print(f"Embedding 输入 Token：{result.embedding_input_tokens}")
    print(f"回退原因：{router_result.fallback_reason or '无'}")
    print("事件：")
    for event in events:
        print(f"- {event.sequence}. {event.step}: {event.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数并返回进程退出码。"""
    try:
        arguments = _parser().parse_args(argv)
    except SystemExit as error:
        if argv is not None and "--help" in argv:
            raise
        return int(error.code)
    if arguments.command == "change-demo":
        from codeinsight.cli.change_demo import run_demo

        return run_demo(arguments.work_dir)
    if arguments.command == "auto-answer":
        return _auto_answer(
            arguments.repo,
            arguments.question,
            arguments.limit,
            arguments.force_route,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
