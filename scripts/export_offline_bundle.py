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
EMBEDDING_ROOT = SERVICE_ROOT / "app" / "embedding"
if str(EMBEDDING_ROOT) not in sys.path:
    sys.path.insert(0, str(EMBEDDING_ROOT))

from bundle import verify_bundle  # noqa: E402

APP_IMAGE_REPOSITORY = "mvi-app-gpu"
DB_IMAGES = ("pgvector/pgvector:pg16", "clickhouse/clickhouse-server:25.8")
DB_IMAGE_AMD64_DIGESTS = {
    "pgvector/pgvector:pg16": "sha256:84a355869251af1a3379cfc9fa7b4dbf962c03f642a4bb7b339a203925071c43",
    "clickhouse/clickhouse-server:25.8": "sha256:cb75da0e596b8115d10bea34cef1414eafebfbcb5f5136c03e022c7b525899b3",
}
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EXPECTED_MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
EXPECTED_SOURCE_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
EXPECTED_PLATFORM = "linux/amd64"
EMBEDDED_MODEL_ROOT = "/opt/mvi-model-bundle"
REQUIRED_FILES = (
    "docker-compose.offline-gpu.yml",
    ".env.offline.example",
    "datasets/video_only_m2ts.yaml",
    "scripts/verify_offline_bundle.py",
    "install-and-start-offline.sh",
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
    return (f"{APP_IMAGE_REPOSITORY}:{git_sha}", *DB_IMAGES)


def pull_database_images(target_platform: str) -> None:
    if target_platform != EXPECTED_PLATFORM:
        raise ValueError(f"database images are pinned only for {EXPECTED_PLATFORM}")
    for image in DB_IMAGES:
        repository = image.rsplit(":", 1)[0]
        pinned_ref = f"{repository}@{DB_IMAGE_AMD64_DIGESTS[image]}"
        run(["docker", "pull", "--platform", target_platform, pinned_ref])
        run(["docker", "tag", pinned_ref, image])


def planned_size_report(git_sha: str) -> dict[str, Any]:
    return {
        "planned_images": list(required_images(git_sha)),
        "target_platform": EXPECTED_PLATFORM,
        "estimated_download_gb": {"min": 7, "max": 11},
        "estimated_build_cache_gb": {"min": 15, "max": 25},
        "estimated_model_bundle_gb": {"min": 4.5, "max": 6.5},
        "estimated_final_transfer_bundle_gb": {"min": 9, "max": 16},
        "recommended_free_disk_gb": 60,
        "estimate_note": "Pre-build conservative range; the model is counted once inside the application image.",
    }


def staging_preflight() -> dict[str, Any]:
    git_head = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    git_status = run(["git", "status", "--short"], cwd=REPO_ROOT)
    if git_status:
        raise RuntimeError("refusing to build a SHA-tagged image from a dirty checkout; commit the release inputs first")
    docker_version = json.loads(run(["docker", "version", "--format", "{{json .}}"]))
    compose_version = run(["docker", "compose", "version", "--short"])
    buildx_version = run(["docker", "buildx", "version"])
    docker_info = json.loads(run(["docker", "info", "--format", "{{json .}}"]))
    return {
        "git_head": git_head,
        "git_status_short": git_status.splitlines(),
        "docker_version": docker_version,
        "compose_version": compose_version,
        "buildx_version": buildx_version,
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


def verify_saved_tar(
    tar_path: Path,
    expected_images: tuple[str, ...],
    *,
    expected_platform: str = EXPECTED_PLATFORM,
) -> dict[str, Any]:
    expected_os, expected_arch = expected_platform.split("/", 1)
    with tarfile.open(tar_path, "r") as archive:
        members = {member.name.lstrip("./"): member for member in archive.getmembers()}
        manifest_member = members.get("manifest.json")
        if manifest_member is None:
            raise ValueError("Docker image archive has no manifest.json")
        handle = archive.extractfile(manifest_member)
        if handle is None:
            raise ValueError("Docker image archive manifest cannot be read")
        records = json.load(handle)
        tags = {tag for record in records for tag in record.get("RepoTags") or []}
        if tags != set(expected_images):
            raise ValueError(
                f"Docker image archive tag set mismatch: expected={sorted(expected_images)}, observed={sorted(tags)}"
            )
        if len(records) != len(expected_images):
            raise ValueError(f"Docker image archive must contain exactly {len(expected_images)} image records")
        image_metadata: dict[str, dict[str, str]] = {}
        for record in records:
            if len(record.get("RepoTags") or []) != 1 or not record.get("Layers"):
                raise ValueError("each Docker image archive record must have exactly one tag and at least one layer")
            required_members = [record.get("Config"), *(record.get("Layers") or [])]
            missing = [name for name in required_members if not name or str(name).lstrip("./") not in members]
            if missing:
                raise ValueError(f"Docker image archive has missing config/layer files: {missing[:10]}")
            config_member = members[str(record["Config"]).lstrip("./")]
            config_handle = archive.extractfile(config_member)
            if config_handle is None:
                raise ValueError(f"Docker image config cannot be read: {record['Config']}")
            config_bytes = config_handle.read()
            config = json.loads(config_bytes)
            observed = f"{str(config.get('os', '')).lower()}/{str(config.get('architecture', '')).lower()}"
            if observed != f"{expected_os.lower()}/{expected_arch.lower()}":
                raise ValueError(
                    f"Docker image archive platform mismatch for {record.get('RepoTags')}: "
                    f"expected {expected_platform}, got {observed}"
                )
            image_id = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
            for tag in record.get("RepoTags") or []:
                image_metadata[tag] = {"image_id": image_id, "platform": observed}
    return {"repo_tags": sorted(tags), "record_count": len(records), "images": image_metadata}


def write_checksums(bundle_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"offline bundle must not contain symlinks: {path}")
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(bundle_root).as_posix()
        entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (bundle_root / "SHA256SUMS").write_text(
        "\n".join(f"{item['sha256']}  {item['path']}" for item in entries) + "\n",
        encoding="utf-8",
    )
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
        env_path.read_text(encoding="utf-8").replace("CHANGE_ME_GIT_SHA", git_sha),
        encoding="utf-8",
    )


def embedded_model_metadata(image_ref: str, verified_model: dict[str, Any], model_bundle: Path) -> dict[str, Any]:
    return {
        "image_ref": image_ref,
        "container_root": EMBEDDED_MODEL_ROOT,
        "model_id": verified_model.get("model_id"),
        "model_revision": verified_model.get("model_revision"),
        "source_commit": verified_model.get("source_commit"),
        "bundle_manifest_sha256": sha256_file(model_bundle / "bundle_manifest.json"),
    }


def build_bundled_application_image(image: str, model_bundle: Path, target_platform: str) -> None:
    run([
        "docker", "buildx", "build", "--platform", target_platform,
        "--build-context", f"mvi_model_bundle={model_bundle}",
        "--target", "gpu-bundled", "--load", "-t", image, ".",
    ], cwd=REPO_ROOT)


def verify_embedded_model_image(image: str, expected_bundle_manifest_sha256: str) -> dict[str, Any]:
    code = (
        "import importlib,json; from pathlib import Path; "
        "from app.embedding.bundle import sha256_file,verify_bundle; "
        f"root=Path({EMBEDDED_MODEL_ROOT!r}); "
        "required=('bundle_manifest.json','model_manifest.json','source_manifest.json'); "
        "assert all((root/name).is_file() for name in required), 'embedded manifests missing'; "
        "assert (root/'model').is_dir() and any(p.is_file() for p in (root/'model').rglob('*')), 'model empty'; "
        "assert (root/'source').is_dir() and any(p.is_file() for p in (root/'source').rglob('*')), 'source empty'; "
        "assert not any(p.is_symlink() for p in root.rglob('*')), 'embedded bundle contains symlink'; "
        f"manifest=verify_bundle(root, expected_model_id={EXPECTED_MODEL_ID!r}, "
        f"expected_model_revision={EXPECTED_MODEL_REVISION!r}, expected_source_commit={EXPECTED_SOURCE_COMMIT!r}); "
        f"assert sha256_file(root/'bundle_manifest.json') == {expected_bundle_manifest_sha256!r}, "
        "'embedded bundle manifest hash mismatch'; "
        "[importlib.import_module(name) for name in ('torch','transformers','qwen_vl_utils')]; "
        "print(json.dumps({'status':'PASS','model_id':manifest['model_id'],"
        "'model_revision':manifest['model_revision'],'source_commit':manifest['source_commit'],"
        "'bundle_manifest_sha256':sha256_file(root/'bundle_manifest.json')}))"
    )
    output = run([
        "docker", "run", "--rm", "--pull", "never", "--network", "none",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", image, "python3", "-c", code,
    ])
    try:
        result = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"embedded model verification returned invalid output: {output!r}") from exc
    expected = {
        "status": "PASS",
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "bundle_manifest_sha256": expected_bundle_manifest_sha256,
    }
    if result != expected:
        raise RuntimeError(f"embedded model verification result mismatch: expected={expected}, observed={result}")
    return result


def gpu_smoke_result(image: str, *, requested: bool) -> dict[str, Any]:
    if not requested:
        return {"status": "NOT_RUN", "detail": "--gpu-smoke was not requested"}
    info = json.loads(run(["docker", "info", "--format", "{{json .}}"]))
    runtimes = {str(name).lower() for name in (info.get("Runtimes") or {})}
    if shutil.which("nvidia-smi") is None or "nvidia" not in runtimes:
        return {"status": "SKIPPED", "detail": "host has no NVIDIA runtime"}
    detail = run([
        "docker", "run", "--rm", "--pull", "never", "--gpus", "all", image,
        "python3", "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
    ])
    return {"status": "PASS", "detail": detail}


def export_bundle(
    output_dir: Path,
    model_bundle: Path,
    *,
    target_platform: str = EXPECTED_PLATFORM,
    git_sha: str | None = None,
    gpu_smoke: bool = False,
) -> dict[str, Any]:
    if target_platform != EXPECTED_PLATFORM:
        raise ValueError(f"the supported institution offline target is exactly {EXPECTED_PLATFORM}")
    sha = git_sha or run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT)
    if not sha or any(character not in "0123456789abcdefABCDEF" for character in sha):
        raise ValueError(f"invalid Git SHA tag: {sha!r}")
    output_dir = output_dir.expanduser().resolve()
    model_bundle = model_bundle.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    preflight = staging_preflight()
    if not preflight["git_head"].startswith(sha):
        raise ValueError(f"Git SHA tag {sha} does not match checked-out HEAD {preflight['git_head']}")
    verified_model = verify_bundle(
        model_bundle,
        expected_model_id=EXPECTED_MODEL_ID,
        expected_model_revision=EXPECTED_MODEL_REVISION,
        expected_source_commit=EXPECTED_SOURCE_COMMIT,
    )
    model_bundle_manifest_sha256 = sha256_file(model_bundle / "bundle_manifest.json")
    images = required_images(sha)
    build_bundled_application_image(images[0], model_bundle, target_platform)
    embedded_model_smoke = verify_embedded_model_image(images[0], model_bundle_manifest_sha256)
    pull_database_images(target_platform)

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
    runtime_smoke = gpu_smoke_result(images[0], requested=gpu_smoke)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mvi-offline-", dir=output_dir.parent) as raw_staging:
        staging = Path(raw_staging) / output_dir.name
        staging.mkdir()
        _copy_contract_files(staging, sha)
        image_dir = staging / "images"
        image_dir.mkdir()
        tar_path = image_dir / "mvi-images-linux-amd64.tar"
        run(["docker", "save", "-o", str(tar_path), *images])
        archive_detail = verify_saved_tar(tar_path, images, expected_platform=target_platform)
        for image in inspected:
            archived = archive_detail["images"].get(image["ref"])
            if not archived or archived["image_id"] != image["image_id"]:
                raise ValueError(f"saved archive image ID mismatch for {image['ref']}")
        deployment_size = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha,
            "build_host": {"system": platform.system(), "machine": platform.machine()},
            "target_platform": target_platform,
            "planned_size_report": planned_size_report(sha),
            "staging_preflight": preflight,
            "gpu_runtime_smoke": runtime_smoke,
            "embedded_model_container_smoke": embedded_model_smoke,
            "images": inspected,
            "image_archive": {
                "path": tar_path.relative_to(staging).as_posix(),
                "size_bytes": tar_path.stat().st_size,
                "sha256": sha256_file(tar_path),
                **archive_detail,
            },
            "embedded_model_bundle": embedded_model_metadata(images[0], verified_model, model_bundle),
            "actual_sizes": {
                "application_image_size_bytes": inspected[0].get("size_bytes"),
                "postgres_image_size_bytes": inspected[1].get("size_bytes"),
                "clickhouse_image_size_bytes": inspected[2].get("size_bytes"),
                "docker_tar_size_bytes": tar_path.stat().st_size,
                "transfer_payload_size_bytes": deployment_size,
                "model_bundle_external_transfer_size_bytes": 0,
                "bundle_directory_size_bytes": 0,
                "docker_system_df_v": run(["docker", "system", "df", "-v"]),
            },
        }
        for _ in range(5):
            (staging / "bundle_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            write_checksums(staging)
            observed_size = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
            if manifest["actual_sizes"]["bundle_directory_size_bytes"] == observed_size:
                break
            manifest["actual_sizes"]["bundle_directory_size_bytes"] = observed_size
        else:
            raise RuntimeError("bundle directory size manifest did not stabilize")
        staging.rename(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one registry-free linux/amd64 MVI image TAR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--target-platform", default=EXPECTED_PLATFORM)
    parser.add_argument("--git-sha", help="optional application tag; defaults to current short Git SHA")
    parser.add_argument("--gpu-smoke", action="store_true", help="run CUDA smoke, or record SKIPPED if unavailable")
    parser.add_argument("--estimate-only", action="store_true", help="print the pre-build size plan and exit")
    args = parser.parse_args()
    sha = args.git_sha or run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT)
    print(json.dumps(planned_size_report(sha), indent=2, sort_keys=True), flush=True)
    if args.estimate_only:
        return 0
    manifest = export_bundle(
        args.output_dir,
        args.model_bundle,
        target_platform=args.target_platform,
        git_sha=sha,
        gpu_smoke=args.gpu_smoke,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
