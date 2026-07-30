#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts
exec > >(tee artifacts/verify_faz7_output.txt) 2>&1

echo "[faz7] verify started: $(date -Iseconds)"
if [[ ! -f .env.faz7 ]]; then
  cp .env.faz7.example .env.faz7
fi

docker compose -f docker-compose.faz7.yml up -d --build

for attempt in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/tmp/faz7_health.json && \
     python -c 'import json; p=json.load(open("/tmp/faz7_health.json")); assert p["status"]=="ok" and p["embedding_mode"]=="synthetic"'; then
    break
  fi
  if [[ "$attempt" == "60" ]]; then
    echo "API did not become healthy" >&2
    exit 1
  fi
  sleep 2
done

cat /tmp/faz7_health.json
docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair

curl -fsS -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"kalabalik trafik","dataset_id":"auair","backend":"clickhouse","strategy":"prefilter","dimension":512,"top_k":10,"repeats":10}'
echo

curl -fsS http://localhost:8000/stats
echo
curl -fsS -o /dev/null http://localhost:7860
echo "[faz7] UI HTTP 200"
echo "[faz7] verify completed: $(date -Iseconds)"

