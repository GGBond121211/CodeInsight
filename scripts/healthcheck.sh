#!/usr/bin/env bash
set -Eeuo pipefail

base_url="${CODEINSIGHT_BASE_URL:-http://127.0.0.1:18000}"
gateway_url="${CODEINSIGHT_GATEWAY_URL:-http://127.0.0.1:18010}"

curl --fail --silent --show-error "${base_url}/api/v1/health" >/dev/null
curl --fail --silent --show-error "${gateway_url}/health" >/dev/null
echo "CodeInsight API and Gateway proxy are healthy."
