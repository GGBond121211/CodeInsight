#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="${CODEINSIGHT_IMAGE_TAG:-2.0.1-local}"
compose=(docker compose -f "$root/ops/docker-compose.yml")

cd "$root"
docker build --file Dockerfile --tag "codeinsight-api:${tag}" .
CODEINSIGHT_IMAGE_TAG="$tag" "${compose[@]}" up -d --remove-orphans
"$root/scripts/healthcheck.sh"
"${compose[@]}" ps
