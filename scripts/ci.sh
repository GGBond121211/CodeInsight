#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/backend"

uv run ruff check src tests
uv run python -m pytest --basetemp="../work/pytest-ci" -q
cd "$root"
docker compose -f ops/docker-compose.yml config >/dev/null
docker build --file Dockerfile --tag codeinsight-api:ci .
