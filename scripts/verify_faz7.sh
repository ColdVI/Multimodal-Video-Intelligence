#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p artifacts
exec > artifacts/verify_faz7_output.txt 2>&1

echo "faz7_verify_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker compose -f docker-compose.faz7.yml config --quiet
docker info >/dev/null

if [[ ! -f .env.faz7 ]]; then
  cp .env.faz7.example .env.faz7
fi

docker compose -f docker-compose.faz7.yml up -d --build

ready=0
for _attempt in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/tmp/faz7_health.json && curl -sf http://localhost:7860/ >/dev/null; then
    ready=1
    break
  fi
  sleep 4
done
if [[ "$ready" != "1" ]]; then
  docker compose -f docker-compose.faz7.yml ps
  docker compose -f docker-compose.faz7.yml logs --tail=200 api ui
  exit 1
fi

cat /tmp/faz7_health.json
docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
curl -sf http://localhost:8000/stats
curl -sf -X POST http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"kalabalik trafik","dataset_id":"auair","backend":"clickhouse","strategy":"prefilter","dimension":512,"top_k":10,"repeats":10}'
curl -sf http://localhost:7860/ >/dev/null
docker compose -f docker-compose.faz7.yml exec -T api pytest service/tests -q
echo "faz7_verify_status=PASS"
