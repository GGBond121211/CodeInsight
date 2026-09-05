# CodeInsight Sandbox

Build the fixed local image once from the repository root:

```powershell
docker build --tag codeinsight-sandbox:py312-v1 ops/sandbox
```

`DockerSandbox` only selects commands from a registered validation profile. Each
container uses `--network none`, a non-root user, CPU/memory/PID limits, a
read-write mount of the temporary workspace, and a bounded wall-clock timeout.
The model never supplies the Docker command or image name.

The image is a local development artifact for Step 6. Step 9 must replace the
floating base image reference with a reviewed digest and register it in the
third-party notice before publication.
