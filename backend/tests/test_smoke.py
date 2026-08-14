"""Smoke tests for package imports and CLI help behavior."""

import importlib

import pytest

from codeinsight.cli.main import main

PACKAGE_NAME = "codeinsight"
CORE_MODULES = (
    "codeinsight.domain",
    "codeinsight.application",
    "codeinsight.ingestion",
    "codeinsight.retrieval",
    "codeinsight.prompts",
    "codeinsight.evaluation",
    "codeinsight.api",
    "codeinsight.infrastructure",
    "codeinsight.cli",
)


def test_package_and_core_modules_importable() -> None:
    assert importlib.import_module(PACKAGE_NAME) is not None
    for module_name in CORE_MODULES:
        assert importlib.import_module(module_name) is not None


def test_cli_main_returns_zero() -> None:
    assert main([]) == 0


def test_cli_help_lists_description_and_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "CodeInsight" in output
    assert "auto-answer" in output
    assert "agent-answer" not in output
