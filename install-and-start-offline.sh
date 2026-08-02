#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$BUNDLE_ROOT/.env.offline"
MODE="start"

usage() {
  printf 'Usage: bash install-and-start-offline.sh [--load-only]\n'
}

case "${1:-}" in
  "") ;;
  --load-only) MODE="load-only" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

command -v python3 >/dev/null 2>&1 || {
  printf 'ERROR: python3 is required to verify this bundle.\n' >&2
  exit 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'ERROR: .env.offline is missing. Create and edit it first:\n' >&2
  printf '  cp .env.offline.example .env.offline\n' >&2
  printf 'Replace all passwords, token, and DATA_ROOT values before retrying.\n' >&2
  exit 1
fi
if grep -q 'CHANGE_ME' "$ENV_FILE"; then
  printf 'ERROR: .env.offline still contains placeholder credentials; refusing to load or start.\n' >&2
  exit 1
fi

VERIFY=(python3 "$BUNDLE_ROOT/scripts/verify_offline_bundle.py" "$BUNDLE_ROOT" --env-file "$ENV_FILE")
if [[ "$MODE" == "load-only" ]]; then
  "${VERIFY[@]}"
  printf 'PASS: checksums, three image IDs/platforms, and embedded model bundle verified; stack was not started.\n'
  exit 0
fi

command -v curl >/dev/null 2>&1 || {
  printf 'ERROR: curl is required for API/UI health checks.\n' >&2
  exit 1
}
"${VERIFY[@]}" --start

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS --max-time 3 "$url" >/dev/null; then
      printf 'PASS: %s is healthy at %s\n' "$name" "$url"
      return 0
    fi
    sleep 5
  done
  printf 'ERROR: %s did not become healthy: %s\n' "$name" "$url" >&2
  return 1
}

wait_for_url "API" "http://127.0.0.1:8000/health"
wait_for_url "UI" "http://127.0.0.1:7860"
docker compose --env-file "$ENV_FILE" -f "$BUNDLE_ROOT/docker-compose.offline-gpu.yml" ps
printf 'PASS: offline MVI stack started with local images only (--no-build --pull never).\n'
