from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_NAMES = {"source_manifest.json", "model_manifest.json", "bundle_manifest.json"}


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    """Return a stable, symlink-free inventory for a bundle subtree."""
    if not root.is_dir():
        raise FileNotFoundError(f"bundle subtree does not exist: {root}")
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"bundle must not contain symlinks: {relative}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": relative.as_posix(), "size_bytes": size, "sha256": sha256_file(path)})
    if not files:
        raise ValueError(f"bundle subtree contains no files: {root}")
    return files, total


def load_bundle_manifest(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "bundle_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bundle manifest at {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported model bundle manifest schema")
    return payload


def verify_bundle(
    bundle_root: Path,
    *,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    manifest = load_bundle_manifest(bundle_root)
    expected = {
        "model_id": expected_model_id,
        "model_revision": expected_model_revision,
        "source_commit": expected_source_commit,
    }
    for field, value in expected.items():
        if value and manifest.get(field) != value:
            raise ValueError(f"bundle {field} mismatch: expected {value}, got {manifest.get(field)}")
    for kind in ("source", "model"):
        detail_path = bundle_root / f"{kind}_manifest.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        observed, total = inventory(bundle_root / kind)
        if observed != detail.get("files") or total != detail.get("total_size_bytes"):
            raise ValueError(f"{kind} bundle hash/size verification failed")
        if manifest.get(f"{kind}_manifest_sha256") != sha256_file(detail_path):
            raise ValueError(f"{kind} manifest hash verification failed")
    return manifest


__all__ = ["inventory", "load_bundle_manifest", "sha256_file", "verify_bundle"]
