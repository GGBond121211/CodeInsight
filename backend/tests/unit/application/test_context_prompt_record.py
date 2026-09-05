from __future__ import annotations

from codeinsight.application.context_assembler import ContextAssembler, ContextRequest
from codeinsight.domain.prompt_models import (
    PromptRelease,
    PromptTemplate,
    PromptVariable,
    PromptVersion,
)
from codeinsight.infrastructure.prompt_store import InMemoryPromptStore


def test_context_assembly_records_prompt_version_without_raw_variables() -> None:
    store = InMemoryPromptStore()
    store.save_template(
        PromptTemplate(
            template_id="template-1",
            name="code-answer",
            scene_code="answer",
            prompt_type="system",
        )
    )
    store.save_version(
        PromptVersion(
            version_id="version-1",
            template_id="template-1",
            version="1.0.0",
            content="安全规则；任务是 {task}",
            variables_schema=(PromptVariable("task", max_length=100),),
        )
    )
    store.save_release(
        PromptRelease(
            release_id="release-1",
            template_id="template-1",
            version_id="version-1",
            environment="local",
        )
    )

    assembly = ContextAssembler(store).assemble(
        ContextRequest(
            system_safety="unused",
            user_goal="解释",
            user_code_task="读取入口",
            prompt_name="code-answer",
            prompt_variables=(("task", "不要泄露密钥"),),
            prompt_run_id="prompt-run-1",
            request_id="request-1",
            run_id="run-1",
            model_version="model-1",
            strategy_version="answer-v1",
        )
    )
    assert assembly.prompt_version is not None
    assert assembly.prompt_version.version_id == "version-1"
    assert assembly.prompt_run is not None
    assert assembly.prompt_run.variables_hash != "不要泄露密钥"
    assert store.get_run("prompt-run-1") == assembly.prompt_run
