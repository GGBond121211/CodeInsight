from subprocess import CompletedProcess

from codeinsight.infrastructure.sandbox import DockerSandbox


def test_preflight_classifies_docker_pipe_permission_failure(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        assert args[0][:2] == ["docker", "version"]
        return CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="open //./pipe/docker_engine: Access is denied",
        )

    monkeypatch.setattr("codeinsight.infrastructure.sandbox.subprocess.run", fake_run)

    result = DockerSandbox().preflight("python_compile")

    assert result.available is False
    assert result.error_class == "SANDBOX_PERMISSION_DENIED"
    assert "Access is denied" in result.message_excerpt


def test_preflight_checks_the_registered_image_after_daemon_is_ready(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return CompletedProcess(args[0], 0, stdout="27.0", stderr="")

    monkeypatch.setattr("codeinsight.infrastructure.sandbox.subprocess.run", fake_run)

    result = DockerSandbox(image="codeinsight-test:latest").preflight("python_compile")

    assert result.available is True
    assert calls == [
        ["docker", "version", "--format", "{{.Server.Version}}"],
        ["docker", "image", "inspect", "codeinsight-test:latest"],
    ]
