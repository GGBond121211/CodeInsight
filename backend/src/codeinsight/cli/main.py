"""CodeInsight 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.application.agent_answer_repository import agent_answer_repository
from codeinsight.application.answer_repository import answer_repository
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.context_assembler import ContextAssembler
from codeinsight.application.query_router import route_question
from codeinsight.application.search_repository import search_repository
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.domain.source import SourceChunk
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
    # 已停用的公开 CLI 入口。下面仍保留辅助实现，因为 Auto Answer 继续复用相同的
    # search、线性回答和 Agent 引擎。
    # search = commands.add_parser("search", help="搜索仓库")
    # answer = commands.add_parser("answer", help="带引用回答仓库问题")
    # agent_answer = commands.add_parser(
    #     "agent-answer", help="通过引用审查和有界修订回答"
    # )
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
    return parser


def _first_line(chunk: SourceChunk) -> str:
    for line in chunk.text.splitlines():
        stripped_line = line.strip()
        if stripped_line:
            return stripped_line
    return ""


def _search(repo: str, question: str, limit: int, retrieval_mode: str) -> int:
    try:
        embedding_model = (
            OpenAIEmbeddingModel.from_environment() if retrieval_mode == "hybrid" else None
        )
        results = search_repository(
            repo,
            question,
            limit=limit,
            retrieval_mode=retrieval_mode,
            semantic_embed=embedding_model.embed if embedding_model else None,
        )
    except ModelConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1
    if not results:
        print("没有找到匹配结果", file=sys.stderr)
        return 1
    for item in results:
        chunk = item.chunk
        print(
            f"{item.rank}\t{item.score}\t{chunk.relative_path}:{chunk.start_line}-"
            f"{chunk.end_line}\t{item.retrieval_reason}\t{_first_line(chunk)}"
        )
    return 0


def _answer(repo: str, question: str, limit: int, retrieval_mode: str) -> int:
    try:
        model = _configured_chat_model()
        embedding_model = (
            OpenAIEmbeddingModel.from_environment() if retrieval_mode == "hybrid" else None
        )
        result = answer_repository(
            repo,
            question,
            generate=model.generate,
            limit=limit,
            retrieval_mode=retrieval_mode,
            semantic_embed=embedding_model.embed if embedding_model else None,
        )
    except ModelConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (ModelCallError, ModelResponseError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(result.answer)
    if result.citations:
        print("证据：")
        for citation in result.citations:
            print(
                f"- [{citation.evidence_id}] {citation.relative_path}:"
                f"{citation.start_line}-{citation.end_line}"
            )
    print(f"检索模式：{result.retrieval_mode}")
    print(f"模型：{result.model or '未调用'}")
    print(f"Prompt：{result.prompt_version}")
    print(f"Token 用量：输入={result.input_tokens or 0} 输出={result.output_tokens or 0}")
    return 0


def _agent_answer(repo: str, question: str, limit: int, retrieval_mode: str) -> int:
    try:
        model = _configured_chat_model()
        embedding_model = (
            OpenAIEmbeddingModel.from_environment() if retrieval_mode == "hybrid" else None
        )
        agent_result = agent_answer_repository(
            repo,
            question,
            complete=model.complete,
            limit=limit,
            retrieval_mode=retrieval_mode,
            semantic_embed=embedding_model.embed if embedding_model else None,
        )
    except ModelConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (ModelCallError, ModelResponseError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    result = agent_result.result
    print(result.answer)
    if result.citations:
        print("证据：")
        for citation in result.citations:
            print(
                f"- [{citation.evidence_id}] {citation.relative_path}:"
                f"{citation.start_line}-{citation.end_line}"
            )
    print("Agent 事件：")
    for event in agent_result.events:
        print(f"- {event.sequence}. {event.step}: {event.summary}")
    print(f"修订次数：{agent_result.revisions}")
    print(f"检索模式：{result.retrieval_mode}")
    print(f"模型：{result.model or '未调用'}")
    print(f"Prompt：{result.prompt_version}")
    print(f"Token 用量：输入={agent_result.input_tokens} 输出={agent_result.output_tokens}")
    return 0


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
    # search / answer / agent-answer 的公开 CLI 分发已停用。
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
