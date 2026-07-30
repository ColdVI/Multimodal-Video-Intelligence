from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.embedding.bundle import inventory, sha256_file, verify_bundle  # noqa: E402


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
DEFAULT_SOURCE_REPO = "https://github.com/QwenLM/Qwen3-VL-Embedding.git"
DEFAULT_SOURCE_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
CRITICAL_PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils", "huggingface-hub")


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, symlinks=False, ignore=shutil.ignore_patterns(".git"))


def _provision_source(target: Path, repo: str, commit: str, local_source: Path | None) -> None:
    if local_source:
        observed = _run(["git", "rev-parse", "HEAD"], cwd=local_source)
        if observed != commit:
            raise ValueError(f"local source commit mismatch: expected {commit}, got {observed}")
        _copy_tree(local_source, target)
        return
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(target)])
    _run(["git", "checkout", "--detach", commit], cwd=target)
    observed = _run(["git", "rev-parse", "HEAD"], cwd=target)
    if observed != commit:
        raise RuntimeError(f"source checkout mismatch: expected {commit}, got {observed}")
    shutil.rmtree(target / ".git")


def _provision_model(target: Path, model_id: str, revision: str, local_model: Path | None) -> None:
    if local_model:
        _copy_tree(local_model, target)
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to download a model bundle") from exc
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=target,
        local_dir_use_symlinks=False,
    )


def _critical_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in CRITICAL_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed_on_bundle_host"
    return versions


def _requirements_contract() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ("service/requirements.txt", "service/requirements-real.txt"):
        path = REPO_ROOT / relative
        result[relative] = sha256_file(path)
    return result


def create_bundle(
    bundle_root: Path,
    *,
    model_id: str,
    model_revision: str,
    source_repo: str,
    source_commit: str,
    source_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    bundle_root = bundle_root.expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle root: {bundle_root}")
    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mvi-bundle-", dir=bundle_root.parent) as raw_temp:
        staging = Path(raw_temp) / "bundle"
        staging.mkdir()
        _provision_source(staging / "source", source_repo, source_commit, source_path)
        _provision_model(staging / "model", model_id, model_revision, model_path)
        generated_at = datetime.now(timezone.utc).isoformat()
        source_files, source_size = inventory(staging / "source")
        model_files, model_size = inventory(staging / "model")
        source_manifest = {
            "schema_version": 1, "kind": "source", "source_repo": source_repo,
            "source_commit": source_commit, "generated_at_utc": generated_at,
            "total_size_bytes": source_size, "files": source_files,
        }
        model_manifest = {
            "schema_version": 1, "kind": "model", "model_id": model_id,
            "model_revision": model_revision, "generated_at_utc": generated_at,
            "total_size_bytes": model_size, "files": model_files,
        }
        for name, payload in (("source_manifest.json", source_manifest), ("model_manifest.json", model_manifest)):
            (staging / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bundle_manifest = {
            "schema_version": 1,
            "generated_at_utc": generated_at,
            "model_id": model_id,
            "model_revision": model_revision,
            "source_repo": source_repo,
            "source_commit": source_commit,
            "source_manifest_sha256": sha256_file(staging / "source_manifest.json"),
            "model_manifest_sha256": sha256_file(staging / "model_manifest.json"),
            "total_size_bytes": source_size + model_size,
            "critical_package_versions": _critical_versions(),
            "requirements_sha256": _requirements_contract(),
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_bundle(
            staging, expected_model_id=model_id, expected_model_revision=model_revision,
            expected_source_commit=source_commit,
        )
        staging.rename(bundle_root)
    return bundle_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a pinned, hash-verified Qwen source/model bundle")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, help="pre-provisioned Git checkout for offline preparation")
    parser.add_argument("--model-path", type=Path, help="pre-provisioned model snapshot for offline preparation")
    args = parser.parse_args()
    manifest = create_bundle(
        args.bundle_root, model_id=args.model_id, model_revision=args.model_revision,
        source_repo=args.source_repo, source_commit=args.source_commit,
        source_path=args.source_path, model_path=args.model_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
