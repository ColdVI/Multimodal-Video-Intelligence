#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/.runtime/offline-bundles}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$REPO_ROOT/.runtime/offline-model-bundle}"
VENV_ROOT="${MVI_BUILDER_VENV:-$REPO_ROOT/.runtime/offline-builder-venv}"
BASE_BRANCH="feat/advanced-retrieval-evidence-gated"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git is required"
BRANCH="$(git branch --show-current)"
GIT_HEAD="$(git rev-parse HEAD)"
GIT_SHA="$(git rev-parse --short=12 HEAD)"
printf 'Repository: %s\nBranch: %s\nCommit: %s\nHost architecture: %s\nTarget platform: %s\n' \
  "$REPO_ROOT" "$BRANCH" "$GIT_HEAD" "$(uname -m)" "$TARGET_PLATFORM"

[[ "$TARGET_PLATFORM" == "linux/amd64" ]] || fail "TARGET_PLATFORM must be exactly linux/amd64"
[[ -z "$(git status --porcelain)" ]] || fail "dirty checkout: commit or stash release inputs before building"

BASE_REF=""
if git show-ref --verify --quiet "refs/heads/$BASE_BRANCH"; then
  BASE_REF="$BASE_BRANCH"
elif git show-ref --verify --quiet "refs/remotes/origin/$BASE_BRANCH"; then
  BASE_REF="origin/$BASE_BRANCH"
fi
[[ -n "$BASE_REF" ]] || fail "cannot find $BASE_BRANCH locally or on origin; refusing an unverified release base"
git merge-base --is-ancestor "$BASE_REF" HEAD || fail "current branch does not descend from $BASE_REF"
printf 'Verified release base: %s\n' "$BASE_REF"

command -v docker >/dev/null 2>&1 || fail "Docker CLI is required"
command -v python3 >/dev/null 2>&1 || fail "Python 3 is required"
docker version >/dev/null || fail "Docker daemon is not running"
docker compose version >/dev/null || fail "Docker Compose v2 is required"
docker buildx version >/dev/null || fail "Docker buildx is required"

mkdir -p "$OUTPUT_ROOT" "$(dirname -- "$MODEL_CACHE_ROOT")" "$(dirname -- "$VENV_ROOT")"
FREE_KB="$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
REQUIRED_KB=$((60 * 1024 * 1024))
[[ "$FREE_KB" =~ ^[0-9]+$ ]] || fail "could not determine free disk space"
(( FREE_KB >= REQUIRED_KB )) || fail "at least 60 GB free disk is required under $OUTPUT_ROOT"

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  python3 -m venv "$VENV_ROOT"
fi
PYTHON_BIN="$VENV_ROOT/bin/python"
if ! "$PYTHON_BIN" -c 'import huggingface_hub' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install 'huggingface-hub==0.36.0'
fi

verify_model_bundle() {
  "$PYTHON_BIN" scripts/prepare_model_bundle.py \
    --bundle-root "$MODEL_CACHE_ROOT" \
    --verify-only >/dev/null
}

if [[ -e "$MODEL_CACHE_ROOT" ]]; then
  verify_model_bundle || fail "existing model bundle is incomplete or corrupt: $MODEL_CACHE_ROOT"
  printf 'Reusing verified model bundle: %s\n' "$MODEL_CACHE_ROOT"
else
  "$PYTHON_BIN" scripts/prepare_model_bundle.py --bundle-root "$MODEL_CACHE_ROOT"
  verify_model_bundle || fail "newly prepared model bundle failed hash verification"
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$OUTPUT_ROOT/mvi-offline-bundle-$GIT_SHA-$TIMESTAMP"
[[ ! -e "$OUTPUT_DIR" ]] || fail "refusing to overwrite output: $OUTPUT_DIR"

"$PYTHON_BIN" scripts/export_offline_bundle.py \
  --model-bundle "$MODEL_CACHE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --target-platform "$TARGET_PLATFORM" \
  --git-sha "$GIT_SHA" \
  --gpu-smoke

"$PYTHON_BIN" "$OUTPUT_DIR/scripts/verify_offline_bundle.py" "$OUTPUT_DIR" --skip-load

printf 'Bundle size:\n'
du -sh "$OUTPUT_DIR"
printf 'Image TAR SHA-256:\n'
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUTPUT_DIR/images/mvi-images-linux-amd64.tar"
else
  sha256sum "$OUTPUT_DIR/images/mvi-images-linux-amd64.tar"
fi
printf 'Offline bundle ready: %s\n' "$OUTPUT_DIR"
