"""有界引用修订 Agent 的应用层入口。"""

from collections.abc import Callable, Sequence
from pathlib import Path

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.domain.agent import AgentRepositoryAnswer
from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.reranker import Reranker

SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]


def agent_answer_repository(
    root: str | Path,
    question: str,
    *,
    complete: Callable[[str, str], ModelCompletion],
    limit: int = 5,
    retrieval_mode: str = "hybrid",
    semantic_embed: SemanticEmbed | None = None,
    reranker: Reranker | None = None,
) -> AgentRepositoryAnswer:
    """通过检索、引用审查和最多五次修订来回答问题。"""
    return run_citation_agent(
        root,
        question,
        complete=complete,
        limit=limit,
        retrieval_mode=retrieval_mode,
        semantic_embed=semantic_embed,
        reranker=reranker,
    )
