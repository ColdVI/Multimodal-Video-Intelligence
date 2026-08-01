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
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EXPECTED_MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
EXPECTED_SOURCE_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
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
    return (f"mvi-app-gpu:{git_sha}", *DB_IMAGES)


def planned_size_report(git_sha: str) -> dict[str, Any]:
    return {
        "planned_images": list(required_images(git_sha)),
        "target_platform": "linux/amd64",
        "estimated_download_gb": {"min": 7, "max": 11},
        "estimated_build_cache_gb": {"min": 15, "max": 25},
        "estimated_model_bundle_gb": {"min": 4.5, "max": 6.5},
        "estimated_final_transfer_bundle_gb": {"min": 13, "max": 20},
        "recommended_free_disk_gb": 60,
        "estimate_note": "Pre-build conservative range; actual TAR size is measured after docker save.",
    }


def staging_preflight(*, allow_dirty: bool = False) -> dict[str, Any]:
    git_head = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    git_status = run(["git", "status", "--short"], cwd=REPO_ROOT)
    if git_status and not allow_dirty:
        raise RuntimeError("refusing to build a SHA-tagged image from a dirty checkout; commit or use --allow-dirty explicitly")
    docker_version = json.loads(run(["docker", "version", "--format", "{{json .}}"]))
    compose_version = run(["docker", "compose", "version", "--short"])
    docker_info = json.loads(run(["docker", "info", "--format", "{{json .}}"]))
    return {
        "git_head": git_head,
        "git_status_short": git_status.splitlines(),
        "docker_version": docker_version,
        "compose_version": compose_version,
        "docker_engine": {
            "os": docker_info.get("OSType"),
            "architecture": docker_info.get("Architecture"),
            "server_version": docker_info.get("ServerVersion"),
        },
    }

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
    allow_dirty: bool = False,
    gpu_smoke: bool = False,
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
    preflight = staging_preflight(allow_dirty=allow_dirty)
    if not preflight["git_head"].startswith(sha):
        raise ValueError(f"Git SHA tag {sha} does not match checked-out HEAD {preflight['git_head']}")
    verified_model = verify_bundle(
        model_bundle, expected_model_id=EXPECTED_MODEL_ID,
        expected_model_revision=EXPECTED_MODEL_REVISION, expected_source_commit=EXPECTED_SOURCE_COMMIT,
    )
    images = required_images(sha)

    run(["docker", "build", "--platform", target_platform, "--target", "gpu", "-t", images[0], "."], cwd=REPO_ROOT)
    for image in DB_IMAGES:
        run(["docker", "pull", "--platform", target_platform, image])
    gpu_runtime_smoke: dict[str, Any] = {"status": "NOT_RUN", "detail": "--gpu-smoke was not requested"}
    if gpu_smoke:
        try:
            detail = run(["docker", "run", "--rm", "--pull", "never", "--gpus", "all", images[0], "python3", "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
            gpu_runtime_smoke = {"status": "PASS", "detail": detail}
        except Exception as exc:
            gpu_runtime_smoke = {"status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}

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
            "size_bytes": detail.get("Size"),
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
            "planned_size_report": planned_size_report(sha),
            "staging_preflight": preflight,
            "gpu_runtime_smoke": gpu_runtime_smoke,
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
                "bundle_manifest_sha256": sha256_file(model_bundle / "bundle_manifest.json"),
            },
            "actual_sizes": {
                "application_image_size_bytes": inspected[0].get("size_bytes"),
                "postgres_image_size_bytes": inspected[1].get("size_bytes"),
                "clickhouse_image_size_bytes": inspected[2].get("size_bytes"),
                "docker_tar_size_bytes": tar_path.stat().st_size,
                "model_bundle_size_bytes": verified_model.get("total_size_bytes"),
                "total_transfer_size_bytes": tar_path.stat().st_size + int(verified_model.get("total_size_bytes") or 0),
                "docker_system_df_v": run(["docker", "system", "df", "-v"]),
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
    parser.add_argument("--allow-dirty", action="store_true", help="explicitly permit a dirty staging checkout")
    parser.add_argument("--gpu-smoke", action="store_true", help="run a non-pulling CUDA smoke with the built app image")
    parser.add_argument("--estimate-only", action="store_true", help="print the pre-build size plan and exit")
    args = parser.parse_args()
    sha = args.git_sha or run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT)
    print(json.dumps(planned_size_report(sha), indent=2, sort_keys=True), flush=True)
    if args.estimate_only:
        return 0
    manifest = export_bundle(
        args.output_dir, args.model_bundle,
        target_platform=args.target_platform, git_sha=sha,
        allow_dirty=args.allow_dirty, gpu_smoke=args.gpu_smoke,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
