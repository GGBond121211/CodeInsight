#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
previous="${CODEINSIGHT_PREVIOUS_IMAGE_TAG:-2.0.0-previous}"

if ! docker image inspect "codeinsight-api:${previous}" >/dev/null 2>&1; then
  echo "Previous local image is unavailable: codeinsight-api:${previous}" >&2
  exit 2
fi

cd "$root"
CODEINSIGHT_IMAGE_TAG="$previous" docker compose -f ops/docker-compose.yml up -d --no-build
"$root/scripts/healthcheck.sh"
