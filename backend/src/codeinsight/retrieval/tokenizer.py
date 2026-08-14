"""Deterministic tokenization for explainable lexical retrieval."""

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+")
_PATH_SEPARATOR = re.compile(r"[\\/]+")
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "after",
        "can",
        "does",
        "do",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
    }
)


def _unique(tokens: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tokens))


def tokenize_terms(text: str) -> tuple[str, ...]:
    """Split text into normalized terms while retaining repeated occurrences."""
    tokens: list[str] = []
    for match in _IDENTIFIER.findall(text):
        lowered = match.lower()
        tokens.append(lowered)
        tokens.extend(part for part in lowered.split("_") if part)
        tokens.extend(part.lower() for part in _CAMEL_PART.findall(match))
    return tuple(tokens)


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into normalized tokens, deduplicated by first occurrence."""
    return _unique(list(tokenize_terms(text)))


def tokenize_path(path: str) -> tuple[str, ...]:
    """Tokenize a repository-relative path honoring both path separators."""
    tokens: list[str] = []
    for component in _PATH_SEPARATOR.split(path):
        if component:
            tokens.extend(tokenize(component))
    return _unique(tokens)


def tokenize_query(text: str) -> tuple[str, ...]:
    """Tokenize a query and remove common question words."""
    return tuple(token for token in tokenize(text) if token not in _QUERY_STOP_WORDS)
