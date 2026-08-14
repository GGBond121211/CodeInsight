"""Explainable BM25 retrieval over SourceChunk instances."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import PurePosixPath

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.tokenizer import tokenize_path, tokenize_query, tokenize_terms

_CODE_EXTENSIONS = frozenset({".py", ".pyi"})
_CODE_BODY_WEIGHT = 2.0
_PATH_HIT_BONUS = 1.0
_FILENAME_HIT_BONUS = 1.0
_IDENTIFIER_HIT_BONUS = 1.0
_SYMBOL_HIT_BONUS = 2.0


def _normalized_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_code_path(path: str) -> bool:
    return PurePosixPath(_normalized_path(path)).suffix.casefold() in _CODE_EXTENSIONS


def _is_whole_identifier(token: str, text: str) -> bool:
    pattern = r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])"
    return re.search(pattern, text.lower()) is not None


def _idf(document_count: int, document_frequency: int) -> float:
    return math.log1p((document_count - document_frequency + 0.5) / (document_frequency + 0.5))


def _body_score(
    query_tokens: tuple[str, ...],
    body_counts: Counter[str],
    body_length: int,
    average_body_length: float,
    document_frequencies: Counter[str],
    document_count: int,
    *,
    k1: float,
    b: float,
) -> float:
    if not body_counts or not query_tokens:
        return 0.0
    normalization = 1.0 - b + b * body_length / average_body_length
    score = 0.0
    for token in query_tokens:
        term_frequency = body_counts.get(token, 0)
        if not term_frequency:
            continue
        inverse_document_frequency = _idf(document_count, document_frequencies[token])
        score += inverse_document_frequency * (
            term_frequency * (k1 + 1.0) / (term_frequency + k1 * normalization)
        )
    return score


def _field_bonuses(query_tokens: tuple[str, ...], chunk: SourceChunk) -> float:
    normalized_path = _normalized_path(chunk.relative_path)
    path_tokens = frozenset(tokenize_path(normalized_path))
    filename_tokens = frozenset(tokenize_terms(PurePosixPath(normalized_path).name))
    symbol_tokens = frozenset(tokenize_terms(chunk.symbol_path or ""))
    score = 0.0
    for token in query_tokens:
        if token in path_tokens:
            score += _PATH_HIT_BONUS
        if token in filename_tokens:
            score += _FILENAME_HIT_BONUS
        if _is_whole_identifier(token, chunk.text):
            score += _IDENTIFIER_HIT_BONUS
        if token in symbol_tokens:
            score += _SYMBOL_HIT_BONUS
    return score


def search_chunks_bm25(
    query: str,
    chunks: Sequence[SourceChunk],
    *,
    limit: int = 5,
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[RankedChunk, ...]:
    """Rank chunks with BM25 body scoring plus code-retrieval field bonuses."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    query_tokens = tokenize_query(query)
    if not query_tokens or not chunks:
        return ()

    documents: list[tuple[SourceChunk, Counter[str], int]] = []
    document_frequencies: Counter[str] = Counter()
    for chunk in chunks:
        body_terms = tokenize_terms(chunk.text)
        body_counts = Counter(body_terms)
        documents.append((chunk, body_counts, len(body_terms)))
        document_frequencies.update(body_counts.keys())

    document_count = len(documents)
    average_body_length = sum(item[2] for item in documents) / document_count
    if average_body_length == 0.0:
        return ()

    scored: list[tuple[float, SourceChunk]] = []
    for chunk, body_counts, body_length in documents:
        body_score = _body_score(
            query_tokens,
            body_counts,
            body_length,
            average_body_length,
            document_frequencies,
            document_count,
            k1=k1,
            b=b,
        )
        if _is_code_path(chunk.relative_path):
            body_score *= _CODE_BODY_WEIGHT
        score = body_score + _field_bonuses(query_tokens, chunk)
        if score > 0.0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].relative_path, item[1].start_line))
    return tuple(
        RankedChunk(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(scored[:limit], start=1)
    )
