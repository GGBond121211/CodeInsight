"""CodeInsight command-line entry point."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.application.agent_answer_repository import agent_answer_repository
from codeinsight.application.answer_repository import answer_repository
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.query_router import route_question
from codeinsight.application.search_repository import search_repository
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.domain.source import SourceChunk
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel

PROJECT_DESCRIPTION = (
    "CodeInsight understands source repositories with verifiable file and line evidence."
)


def _positive_int(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return limit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeinsight", description=PROJECT_DESCRIPTION)
    commands = parser.add_subparsers(dest="command")
    # Disabled public CLI entries. Their helper implementations remain below because
    # Auto Answer still reuses the same search, linear-answer, and Agent engines.
    # search = commands.add_parser("search", help="search a repository")
    # answer = commands.add_parser("answer", help="answer a repository question with citations")
    # agent_answer = commands.add_parser(
    #     "agent-answer", help="answer with citation review and bounded revision"
    # )
    auto_answer = commands.add_parser(
        "auto-answer", help="route a multilingual question and answer with public plan metadata"
    )
    auto_answer.add_argument("--repo", required=True, metavar="PATH")
    auto_answer.add_argument("question", metavar="QUESTION")
    auto_answer.add_argument("--limit", type=_positive_int, default=5, metavar="N")
    auto_answer.add_argument(
        "--force-route",
        choices=("linear", "agent"),
        default=None,
        dest="force_route",
        help="override the router's public execution route",
    )
    return parser


def _first_line(chunk: SourceChunk) -> str:
    return next((line.strip() for line in chunk.text.splitlines() if line.strip()), "")


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
        print("no matching results found", file=sys.stderr)
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
        model = OpenAIChatModel.from_environment()
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
        print("evidence:")
        for citation in result.citations:
            print(
                f"- [{citation.evidence_id}] {citation.relative_path}:"
                f"{citation.start_line}-{citation.end_line}"
            )
    print(f"retrieval: {result.retrieval_mode}")
    print(f"model: {result.model or 'not-called'}")
    print(f"prompt: {result.prompt_version}")
    print(f"tokens: input={result.input_tokens or 0} output={result.output_tokens or 0}")
    return 0


def _agent_answer(repo: str, question: str, limit: int, retrieval_mode: str) -> int:
    try:
        model = OpenAIChatModel.from_environment()
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
        print("evidence:")
        for citation in result.citations:
            print(
                f"- [{citation.evidence_id}] {citation.relative_path}:"
                f"{citation.start_line}-{citation.end_line}"
            )
    print("agent-events:")
    for event in agent_result.events:
        print(f"- {event.sequence}. {event.step}: {event.summary}")
    print(f"revisions: {agent_result.revisions}")
    print(f"retrieval: {result.retrieval_mode}")
    print(f"model: {result.model or 'not-called'}")
    print(f"prompt: {result.prompt_version}")
    print(f"tokens: input={agent_result.input_tokens} output={agent_result.output_tokens}")
    return 0


def _auto_answer(repo: str, question: str, limit: int, force_route: str | None) -> int:
    try:
        model = OpenAIChatModel.from_environment()
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
        if router_result.plan.execution_route == "agent":
            agent_result = run_citation_agent(
                repo,
                question,
                complete=model.complete,
                limit=limit,
                retrieval_mode="auto",
                semantic_embed=embedding_model.embed if embedding_model else None,
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
        print("evidence:")
        for citation in result.citations:
            print(
                f"- [{citation.evidence_id}] {citation.relative_path}:"
                f"{citation.start_line}-{citation.end_line}"
            )
    print(f"route: {router_result.plan.execution_route}")
    print(
        f"router: model={router_result.model or 'not-called'} "
        f"tokens={router_result.input_tokens}/{router_result.output_tokens} "
        f"elapsed_ms={router_result.elapsed_milliseconds:.1f}"
    )
    print(f"embedding_input_tokens: {result.embedding_input_tokens}")
    print(f"fallback: {router_result.fallback_reason or 'none'}")
    print("events:")
    for event in events:
        print(f"- {event.sequence}. {event.step}: {event.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and return the process exit code."""
    try:
        arguments = _parser().parse_args(argv)
    except SystemExit as error:
        if argv is not None and "--help" in argv:
            raise
        return int(error.code)
    # Disabled public CLI dispatch for search / answer / agent-answer.
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
