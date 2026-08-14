"""确定性检索分词测试。"""

from codeinsight.retrieval.tokenizer import tokenize, tokenize_path


def test_tokenize_splits_identifiers_and_lowercases() -> None:
    assert tokenize("get_user_by_id") == ("get_user_by_id", "get", "user", "by", "id")
    assert tokenize("getUserById") == ("getuserbyid", "get", "user", "by", "id")
    assert tokenize("UserProfile") == ("userprofile", "user", "profile")
    assert tokenize("HTTPClient") == ("httpclient", "http", "client")


def test_tokenize_deduplicates_and_keeps_numbers() -> None:
    assert tokenize("Foo_foo FOO foo") == ("foo_foo", "foo")
    assert tokenize("year2024 version 2") == ("year2024", "year", "2024", "version", "2")


def test_tokenize_path_handles_separators_and_deduplicates() -> None:
    assert tokenize_path("src/shop/config.py") == ("src", "shop", "config", "py")
    assert tokenize_path("src\\src\\main.py") == ("src", "main", "py")


def test_tokenize_empty_or_punctuation_only() -> None:
    assert tokenize("") == ()
    assert tokenize("!!! ??? --") == ()
    assert tokenize_path("") == ()
