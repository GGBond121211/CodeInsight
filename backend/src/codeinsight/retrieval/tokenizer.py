"""用于可解释词法检索的确定性分词。"""

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
    """把文本切分为规范化词项，同时保留重复出现的词项。"""
    tokens: list[str] = []
    for match in _IDENTIFIER.findall(text):
        lowered = match.lower()
        tokens.append(lowered)
        tokens.extend(part for part in lowered.split("_") if part)
        tokens.extend(part.lower() for part in _CAMEL_PART.findall(match))
    return tuple(tokens)


def tokenize(text: str) -> tuple[str, ...]:
    """把文本切分为规范化 token，并按首次出现顺序去重。"""
    return _unique(list(tokenize_terms(text)))


def tokenize_path(path: str) -> tuple[str, ...]:
    """对仓库相对路径分词，同时兼容两种路径分隔符。"""
    tokens: list[str] = []
    for component in _PATH_SEPARATOR.split(path):
        if component:
            tokens.extend(tokenize(component))
    return _unique(tokens)


def tokenize_query(text: str) -> tuple[str, ...]:
    """对查询分词，并移除常见疑问词。"""
    return tuple(token for token in tokenize(text) if token not in _QUERY_STOP_WORDS)
