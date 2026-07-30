#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts artifacts/research
exec > >(tee artifacts/verify_faz8_output.txt) 2>&1

echo "[faz8] verify started: $(date -Iseconds)"
[[ -f .env.faz7 ]] || cp .env.faz7.example .env.faz7
docker compose -f docker-compose.faz7.yml up -d --build
docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
python scripts/readiness_check.py --profile system --strict
RUN_FAZ8_INTEGRATION=1 python -m pytest -q service/tests -p no:cacheprovider
PYTHONPATH=service python -m app.bench.matrix --suite all --quick --out artifacts/research/test_matrix_all.csv
python scripts/readiness_check.py --profile quality --json
echo "[faz8] quality readiness is informational until Colab artifacts and cached ingest exist"
echo "[faz8] verify completed: $(date -Iseconds)"
