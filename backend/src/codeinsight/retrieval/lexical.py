"""对 SourceChunk 执行可解释的词法评分。"""

import re
from collections.abc import Sequence
from pathlib import PurePosixPath

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.tokenizer import tokenize, tokenize_path, tokenize_query

_CODE_EXTENSIONS = frozenset({".py", ".pyi"})
_BODY_CODE_HIT_WEIGHT = 2.0
_BODY_HIT_WEIGHT = 1.0
_PATH_HIT_WEIGHT = 1.0
_FILENAME_HIT_BONUS = 1.0
_IDENTIFIER_HIT_BONUS = 1.0


def _normalized_path(path: str) -> str:
    return path.replace("\\", "/")


def _body_hit_weight(relative_path: str) -> float:
    suffix = PurePosixPath(_normalized_path(relative_path)).suffix.casefold()
    return _BODY_CODE_HIT_WEIGHT if suffix in _CODE_EXTENSIONS else _BODY_HIT_WEIGHT


def _is_whole_identifier(token: str, text: str) -> bool:
    pattern = r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])"
    return re.search(pattern, text.lower()) is not None


def score_chunk(query_tokens: tuple[str, ...], chunk: SourceChunk) -> float:
    """根据正文、路径、文件名和标识符命中情况为一个块评分。"""
    if not query_tokens:
        return 0.0
    body_tokens = frozenset(tokenize(chunk.text))
    normalized_path = _normalized_path(chunk.relative_path)
    path_tokens = frozenset(tokenize_path(normalized_path))
    filename_tokens = frozenset(tokenize(PurePosixPath(normalized_path).name))
    body_weight = _body_hit_weight(chunk.relative_path)
    score = 0.0
    for token in query_tokens:
        if token in body_tokens:
            score += body_weight
            if _is_whole_identifier(token, chunk.text):
                score += _IDENTIFIER_HIT_BONUS
        if token in path_tokens:
            score += _PATH_HIT_WEIGHT
        if token in filename_tokens:
            score += _FILENAME_HIT_BONUS
    return score


def search_chunks(
    query: str,
    chunks: Sequence[SourceChunk],
    *,
    limit: int = 5,
) -> tuple[RankedChunk, ...]:
    """确定性地为块排序，并丢弃分数为 0 的结果。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    query_tokens = tokenize_query(query)
    if not query_tokens:
        return ()
    scored: list[tuple[float, SourceChunk]] = []
    for chunk in chunks:
        score = score_chunk(query_tokens, chunk)
        if score > 0.0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].relative_path, item[1].start_line))
    return tuple(
        RankedChunk(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(scored[:limit], start=1)
    )
