# Pinned Qwen model bundle

Faz 11 uses two immutable upstream identifiers:

- model: `Qwen/Qwen3-VL-Embedding-2B` at revision
  `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`
- source: `https://github.com/QwenLM/Qwen3-VL-Embedding.git` at commit
  `393e2978d27852b0d0230d6994f37f9c15bed73c`

The source commit was resolved from the official repository's `HEAD` with
`git ls-remote` on 2026-07-30. A branch or mutable `main` reference is not used
by the runtime contract.

## Prepare on a connected host

```bash
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root /opt/mvi-model-bundle
```

The command refuses to overwrite an existing target. It creates `source/`,
`model/`, `source_manifest.json`, `model_manifest.json`, and
`bundle_manifest.json`. Every regular file has a recorded size and SHA-256;
source/model revisions and critical package versions are also recorded.

For a disconnected preparation station, point `--source-path` at a checkout
whose `HEAD` is the pinned commit and `--model-path` at a provisioned snapshot.
The same inventories and verification are applied. Copy the completed bundle
without changing its contents.

## Deploy and verify

Set `MODEL_BUNDLE_ROOT` to the host bundle path and use the additive GPU
override:

```bash
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml exec -T api \
  python -m app.preflight --dataset /workspace/datasets/kurum.yaml
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml exec -T api \
  python /app/scripts/gpu_smoke.py --dataset /workspace/datasets/kurum.yaml \
  --data-root /workspace/data --output /workspace/artifacts/faz11/gpu_smoke.json
```

The Compose profile mounts the bundle read-only. The GPU image performs no Git
clone and no model download during build. `sdpa` is the portable default;
`flash_attention_2` is an explicitly selected optional acceleration profile.

The model/source bundle alone does **not** make the whole deployment air-gapped.
An offline site must also pre-provision Docker base/service images, Python
wheels or an approved package mirror, PostgreSQL/ClickHouse images, the NVIDIA
driver, and NVIDIA Container Toolkit.
