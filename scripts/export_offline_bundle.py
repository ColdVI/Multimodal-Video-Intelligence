from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.embedding.bundle import verify_bundle  # noqa: E402

DB_IMAGES = ("pgvector/pgvector:pg16", "clickhouse/clickhouse-server:25.8")
REQUIRED_FILES = (
    "Dockerfile",
    "requirements.txt",
    "docker-compose.offline-gpu.yml",
    ".env.offline.example",
    "datasets/video_only_m2ts.yaml",
    "scripts/verify_offline_bundle.py",
)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def required_images(git_sha: str) -> tuple[str, ...]:
    return (f"mvi-api-gpu:{git_sha}", f"mvi-ui:{git_sha}", *DB_IMAGES)


def image_inspect(image: str) -> dict[str, Any]:
    payload = json.loads(run(["docker", "image", "inspect", image]))
    if not payload:
        raise RuntimeError(f"docker inspect returned no record for {image}")
    return payload[0]


def verify_image_platform(image: str, detail: dict[str, Any], target_platform: str) -> None:
    target_os, target_arch = target_platform.split("/", 1)
    observed = f"{str(detail.get('Os', '')).lower()}/{str(detail.get('Architecture', '')).lower()}"
    if observed != f"{target_os.lower()}/{target_arch.lower()}":
        raise ValueError(f"image platform mismatch for {image}: expected {target_platform}, got {observed}")


def verify_saved_tar(tar_path: Path, expected_images: tuple[str, ...]) -> dict[str, Any]:
    with tarfile.open(tar_path, "r") as archive:
        names = {member.name.lstrip("./") for member in archive.getmembers()}
        try:
            manifest_member = next(member for member in archive.getmembers() if member.name.lstrip("./") == "manifest.json")
        except StopIteration as exc:
            raise ValueError("Docker image archive has no manifest.json") from exc
        handle = archive.extractfile(manifest_member)
        if handle is None:
            raise ValueError("Docker image archive manifest cannot be read")
        records = json.load(handle)
        tags = {tag for record in records for tag in record.get("RepoTags") or []}
        missing = sorted(set(expected_images) - tags)
        if missing:
            raise ValueError(f"Docker image archive is missing tags: {missing}")
        missing_layers = sorted(
            layer for record in records for layer in record.get("Layers") or []
            if layer.lstrip("./") not in names
        )
        if missing_layers:
            raise ValueError(f"Docker image archive has missing parent/layer files: {missing_layers[:10]}")
    return {"repo_tags": sorted(tags), "record_count": len(records)}


def write_checksums(bundle_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        if path.is_symlink():
            raise ValueError(f"offline bundle must not contain symlinks: {path}")
        relative = path.relative_to(bundle_root).as_posix()
        entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    lines = [f"{item['sha256']}  {item['path']}" for item in entries]
    (bundle_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return entries


def _copy_contract_files(target: Path, git_sha: str) -> None:
    for relative in REQUIRED_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required bundle input is missing: {source}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    env_path = target / ".env.offline.example"
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        .replace("CHANGE_ME_GIT_SHA", git_sha),
        encoding="utf-8",
    )


def export_bundle(
    output_dir: Path,
    model_bundle: Path,
    *,
    target_platform: str = "linux/amd64",
    git_sha: str | None = None,
) -> dict[str, Any]:
    if target_platform != "linux/amd64":
        raise ValueError("the supported institution offline target is exactly linux/amd64")
    sha = git_sha or run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT)
    if not sha or any(character not in "0123456789abcdefABCDEF" for character in sha):
        raise ValueError(f"invalid Git SHA tag: {sha!r}")
    output_dir = output_dir.expanduser().resolve()
    model_bundle = model_bundle.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    verified_model = verify_bundle(model_bundle)
    images = required_images(sha)

    run(["docker", "build", "--platform", target_platform, "--target", "gpu", "-t", images[0], "."], cwd=REPO_ROOT)
    run(["docker", "build", "--platform", target_platform, "--target", "ui", "-t", images[1], "."], cwd=REPO_ROOT)
    for image in DB_IMAGES:
        run(["docker", "pull", "--platform", target_platform, image])

    inspected: list[dict[str, Any]] = []
    for image in images:
        detail = image_inspect(image)
        verify_image_platform(image, detail, target_platform)
        inspected.append({
            "ref": image,
            "image_id": detail.get("Id"),
            "repo_digests": sorted(detail.get("RepoDigests") or []),
            "os": detail.get("Os"),
            "architecture": detail.get("Architecture"),
        })

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mvi-offline-", dir=output_dir.parent) as raw_staging:
        staging = Path(raw_staging) / output_dir.name
        staging.mkdir()
        _copy_contract_files(staging, sha)
        shutil.copytree(model_bundle, staging / "model-bundle", symlinks=False)
        image_dir = staging / "images"
        image_dir.mkdir()
        tar_path = image_dir / "mvi-images-linux-amd64.tar"
        run(["docker", "save", "-o", str(tar_path), *images])
        archive_detail = verify_saved_tar(tar_path, images)
        generated = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": 1,
            "generated_at_utc": generated,
            "git_sha": sha,
            "build_host": {"system": platform.system(), "machine": platform.machine()},
            "target_platform": target_platform,
            "images": inspected,
            "image_archive": {
                "path": tar_path.relative_to(staging).as_posix(),
                "size_bytes": tar_path.stat().st_size,
                "sha256": sha256_file(tar_path),
                **archive_detail,
            },
            "model_bundle": {
                "path": "model-bundle",
                "model_id": verified_model.get("model_id"),
                "model_revision": verified_model.get("model_revision"),
                "source_commit": verified_model.get("source_commit"),
                "total_size_bytes": verified_model.get("total_size_bytes"),
            },
            "requirements": {
                "path": "requirements.txt",
                "sha256": sha256_file(staging / "requirements.txt"),
            },
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_checksums(staging)
        staging.rename(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the complete registry-free linux/amd64 MVI bundle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--target-platform", default="linux/amd64")
    parser.add_argument("--git-sha", help="optional application tag; defaults to current short Git SHA")
    args = parser.parse_args()
    manifest = export_bundle(
        args.output_dir, args.model_bundle,
        target_platform=args.target_platform, git_sha=args.git_sha,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
