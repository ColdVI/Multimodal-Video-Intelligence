from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_SERVICES = {"pg", "ch", "api", "ui"}
EXPECTED_PLATFORM = "linux/amd64"
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EXPECTED_MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
EXPECTED_SOURCE_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
EMBEDDED_MODEL_ROOT = "/opt/mvi-model-bundle"
DB_IMAGES = {"pgvector/pgvector:pg16", "clickhouse/clickhouse-server:25.8"}
REQUIRED_TRANSFER_FILES = {
    ".env.offline.example",
    "bundle_manifest.json",
    "datasets/video_only_m2ts.yaml",
    "docker-compose.offline-gpu.yml",
    "images/mvi-images-linux-amd64.tar",
    "install-and-start-offline.sh",
    "scripts/verify_offline_bundle.py",
}


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


def read_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid SHA256SUMS line {line_number}") from exc
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe SHA256SUMS path: {relative}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            raise ValueError(f"invalid SHA-256 at line {line_number}")
        if relative in checksums:
            raise ValueError(f"duplicate SHA256SUMS path: {relative}")
        checksums[relative] = digest.lower()
    if not checksums:
        raise ValueError("SHA256SUMS is empty")
    return checksums


def verify_checksums(bundle_root: Path) -> dict[str, str]:
    expected = read_sha256sums(bundle_root / "SHA256SUMS")
    actual_files: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"offline bundle must not contain symlinks: {path.relative_to(bundle_root)}")
        if path.is_file() and path.name != "SHA256SUMS":
            actual_files.add(path.relative_to(bundle_root).as_posix())
    if actual_files != set(expected):
        raise ValueError(
            f"bundle inventory mismatch: missing={sorted(set(expected) - actual_files)}, "
            f"unexpected={sorted(actual_files - set(expected))}"
        )
    for relative, digest in expected.items():
        observed = sha256_file(bundle_root / Path(relative))
        if observed != digest:
            raise ValueError(f"checksum mismatch for {relative}: expected {digest}, got {observed}")
    return expected


def inspect_saved_tar(tar_path: Path, expected_images: set[str]) -> dict[str, dict[str, str]]:
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
        if tags != expected_images or len(records) != len(expected_images):
            raise ValueError(
                f"Docker image archive must contain exactly the three required tags: "
                f"expected={sorted(expected_images)}, observed={sorted(tags)}"
            )
        result: dict[str, dict[str, str]] = {}
        for record in records:
            if len(record.get("RepoTags") or []) != 1 or not record.get("Layers"):
                raise ValueError("each Docker image archive record must have exactly one tag and at least one layer")
            filenames = [record.get("Config"), *(record.get("Layers") or [])]
            missing = [name for name in filenames if not name or str(name).lstrip("./") not in members]
            if missing:
                raise ValueError(f"Docker image archive has missing config/layer files: {missing[:10]}")
            config_handle = archive.extractfile(members[str(record["Config"]).lstrip("./")])
            if config_handle is None:
                raise ValueError(f"Docker image config cannot be read: {record['Config']}")
            config_bytes = config_handle.read()
            config = json.loads(config_bytes)
            observed_platform = f"{str(config.get('os', '')).lower()}/{str(config.get('architecture', '')).lower()}"
            if observed_platform != EXPECTED_PLATFORM:
                raise ValueError(
                    f"Docker image archive platform mismatch for {record.get('RepoTags')}: "
                    f"expected {EXPECTED_PLATFORM}, got {observed_platform}"
                )
            image_id = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
            for tag in record.get("RepoTags") or []:
                result[tag] = {"image_id": image_id, "platform": observed_platform}
    return result


def validate_compose_contract(compose_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    text = compose_path.read_text(encoding="utf-8")
    section = re.search(r"(?ms)^services:\s*$\n(.*?)(?=^[A-Za-z][A-Za-z0-9_-]*:\s*$|\Z)", text)
    if section is None:
        raise ValueError("offline compose has no services section")
    body = section.group(1)
    headings = list(re.finditer(r"(?m)^  ([A-Za-z0-9_-]+):\s*$", body))
    blocks: dict[str, str] = {}
    for index, match in enumerate(headings):
        stop = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        blocks[match.group(1)] = body[match.end():stop]
    if set(blocks) != EXPECTED_SERVICES:
        raise ValueError(f"offline compose services must be exactly {sorted(EXPECTED_SERVICES)}")
    images: dict[str, str] = {}
    for name, block in blocks.items():
        if re.search(r"(?m)^    build:\s*", block):
            raise ValueError(f"offline service {name} contains a forbidden build section")
        if not re.search(r"(?m)^    pull_policy:\s*never\s*$", block):
            raise ValueError(f"offline service {name} must set pull_policy: never")
        match = re.search(r"(?m)^    image:\s*(.+?)\s*$", block)
        if match is None:
            raise ValueError(f"offline service {name} has no explicit image")
        images[name] = match.group(1)
        if images[name].endswith(":latest"):
            raise ValueError(f"offline service {name} uses an unpinned latest tag")
    api = blocks["api"]
    ui = blocks["ui"]
    expected_template = "mvi-app-gpu:${MVI_IMAGE_TAG:?MVI_IMAGE_TAG must be set}"
    if images["api"] != images["ui"] or images["api"] != expected_template:
        raise ValueError("offline API and UI must use the same SHA-variable mvi-app-gpu image")
    if re.search(r"(?m)^    command:\s*", api):
        raise ValueError("offline API must preserve the application image uvicorn command")
    if re.search(r"(?m)^    gpus:\s*", ui):
        raise ValueError("offline UI must not request a GPU")
    if not re.search(r'(?m)^    command:\s*\["python3", "-m", "ui\.app"\]\s*$', ui):
        raise ValueError("offline UI must start with python3 -m ui.app")
    if images["pg"] != "pgvector/pgvector:pg16" or images["ch"] != "clickhouse/clickhouse-server:25.8":
        raise ValueError("offline database image tags do not match the pinned contract")
    if not re.search(r"(?m)^    gpus:\s*all\s*$", api):
        raise ValueError("offline API must request all NVIDIA GPUs")
    required_environment = {
        "ENABLED_VECTOR_BACKENDS": "clickhouse",
        "DEFAULT_VECTOR_BACKEND": "clickhouse",
        "ENABLED_DIMENSIONS": '"512"',
        "EMBEDDING_MODE": "real",
        "MODEL_BUNDLE_ROOT": EMBEDDED_MODEL_ROOT,
        "QWEN_REPO_PATH": f"{EMBEDDED_MODEL_ROOT}/source",
        "QWEN_MODEL_PATH": f"{EMBEDDED_MODEL_ROOT}/model",
    }
    for key, expected in required_environment.items():
        if not re.search(rf"(?m)^      {re.escape(key)}:\s*{re.escape(expected)}\s*$", api):
            raise ValueError(f"offline API {key} must be {expected!r}")
    service_volumes: dict[str, str] = {}
    for name, block in blocks.items():
        match = re.search(r"(?ms)^    volumes:\s*$\n(.*?)(?=^    [A-Za-z][A-Za-z0-9_-]*:\s*|\Z)", block)
        service_volumes[name] = match.group(1) if match else ""
        if "MODEL_BUNDLE_ROOT" in service_volumes[name] or ":/opt/mvi-model-bundle" in service_volumes[name]:
            raise ValueError("offline Compose must not mount a host model bundle")
    if "/workspace/data:ro" not in service_volumes["api"]:
        raise ValueError("offline API must mount the external video dataset read-only")

    manifest_images = manifest.get("images", [])
    manifest_refs = {item.get("ref") for item in manifest_images}
    git_sha = manifest.get("git_sha")
    expected_refs = {f"mvi-app-gpu:{git_sha}", *DB_IMAGES}
    if len(manifest_images) != 3 or manifest_refs != expected_refs:
        raise ValueError(
            f"bundle manifest image set must contain exactly three pinned images: "
            f"expected={sorted(expected_refs)}, observed={sorted(manifest_refs)}"
        )
    return {"services": {name: {"image": image} for name, image in images.items()}}


def _verify_embedded_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("embedded_model_bundle")
    if not isinstance(metadata, dict):
        raise ValueError("offline bundle schema v2 requires embedded_model_bundle metadata")
    expected = {
        "image_ref": f"mvi-app-gpu:{manifest.get('git_sha')}",
        "container_root": EMBEDDED_MODEL_ROOT,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"embedded model {field} mismatch: expected {value}, got {metadata.get(field)}")
    digest = str(metadata.get("bundle_manifest_sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise ValueError("embedded model bundle_manifest_sha256 is invalid")
    return metadata


def verify_static_bundle(bundle_root: Path) -> dict[str, Any]:
    bundle_root = bundle_root.expanduser().resolve()
    if (bundle_root / "model-bundle").exists():
        raise ValueError("schema v2 forbids a separate model-bundle transfer directory")
    checksums = verify_checksums(bundle_root)
    missing_transfer_files = sorted(REQUIRED_TRANSFER_FILES - set(checksums))
    if missing_transfer_files:
        raise ValueError(f"offline bundle is missing required transfer files: {missing_transfer_files}")
    manifest = json.loads((bundle_root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(
            f"unsupported offline bundle schema {manifest.get('schema_version')!r}; expected schema_version 2"
        )
    if manifest.get("target_platform") != EXPECTED_PLATFORM:
        raise ValueError(f"offline target must be {EXPECTED_PLATFORM}")
    compose = validate_compose_contract(bundle_root / "docker-compose.offline-gpu.yml", manifest)
    embedded = _verify_embedded_metadata(manifest)
    if (manifest.get("embedded_model_container_smoke") or {}).get("status") != "PASS":
        raise ValueError("offline bundle lacks a successful embedded model container smoke record")
    archive = bundle_root / str((manifest.get("image_archive") or {}).get("path", ""))
    if archive.stat().st_size != manifest["image_archive"].get("size_bytes"):
        raise ValueError("image archive size does not match bundle manifest")
    archive_relative = archive.relative_to(bundle_root).as_posix()
    if checksums.get(archive_relative) != manifest["image_archive"].get("sha256"):
        raise ValueError("image archive hash does not match bundle manifest")
    sizes = manifest.get("actual_sizes") or {}
    if sizes.get("model_bundle_external_transfer_size_bytes") != 0:
        raise ValueError("schema v2 must not count an external model bundle in transfer size")
    payload_size = sum(
        path.stat().st_size for path in bundle_root.rglob("*")
        if path.is_file() and path.name not in {"bundle_manifest.json", "SHA256SUMS"}
    )
    if sizes.get("transfer_payload_size_bytes") != payload_size:
        raise ValueError("transfer payload size does not match bundle manifest")
    directory_size = sum(path.stat().st_size for path in bundle_root.rglob("*") if path.is_file())
    if sizes.get("bundle_directory_size_bytes") != directory_size:
        raise ValueError("bundle directory size does not match bundle manifest")
    expected_refs = {item["ref"] for item in manifest["images"]}
    archived = inspect_saved_tar(archive, expected_refs)
    manifest_images = {item["ref"]: item for item in manifest["images"]}
    for ref, detail in archived.items():
        recorded = manifest_images[ref]
        if detail["image_id"] != recorded.get("image_id"):
            raise ValueError(f"image archive ID does not match bundle manifest for {ref}")
        observed = f"{str(recorded.get('os', '')).lower()}/{str(recorded.get('architecture', '')).lower()}"
        if observed != EXPECTED_PLATFORM:
            raise ValueError(f"bundle manifest image platform mismatch for {ref}: {observed}")
    return {
        "manifest": manifest,
        "compose": compose,
        "embedded_model_bundle": embedded,
        "verified_file_count": len(checksums),
    }


def verify_runtime() -> dict[str, Any]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker CLI is not installed")
    docker_version = run(["docker", "version", "--format", "{{json .Server}}"])
    compose_version = run(["docker", "compose", "version", "--short"])
    info = json.loads(run(["docker", "info", "--format", "{{json .}}"]))
    observed = f"{str(info.get('OSType', '')).lower()}/{str(info.get('Architecture', '')).lower()}"
    if observed != EXPECTED_PLATFORM:
        raise RuntimeError(f"Docker engine platform must be {EXPECTED_PLATFORM}, got {observed}")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not installed or is not on PATH")
    nvidia = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    runtimes = info.get("Runtimes") or {}
    if "nvidia" not in {str(name).lower() for name in runtimes}:
        raise RuntimeError("Docker NVIDIA Container Toolkit runtime is unavailable")
    return {
        "docker_server": json.loads(docker_version),
        "compose_version": compose_version,
        "engine_platform": observed,
        "nvidia_smi": nvidia.splitlines(),
    }


def verify_local_images(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for image in manifest["images"]:
        try:
            payload = json.loads(run(["docker", "image", "inspect", image["ref"]]))[0]
        except Exception as exc:
            raise RuntimeError(
                f"required offline image is missing from the local store: {image['ref']}; refusing to pull"
            ) from exc
        observed = f"{str(payload.get('Os', '')).lower()}/{str(payload.get('Architecture', '')).lower()}"
        if observed != EXPECTED_PLATFORM:
            raise RuntimeError(f"loaded image {image['ref']} has wrong platform: {observed}")
        if payload.get("Id") != image.get("image_id"):
            raise RuntimeError(f"loaded image ID mismatch for {image['ref']}")
        result.append({"ref": image["ref"], "image_id": payload.get("Id"), "platform": observed})
    return result


def verify_embedded_model_image(manifest: dict[str, Any]) -> dict[str, Any]:
    image = manifest["embedded_model_bundle"]["image_ref"]
    code = (
        "import importlib,json; from pathlib import Path; "
        "from app.embedding.bundle import sha256_file,verify_bundle; "
        f"root=Path({EMBEDDED_MODEL_ROOT!r}); "
        "required=('bundle_manifest.json','model_manifest.json','source_manifest.json'); "
        "assert all((root/name).is_file() for name in required), 'embedded manifests missing'; "
        "assert (root/'model').is_dir() and any(p.is_file() for p in (root/'model').rglob('*')), 'model empty'; "
        "assert (root/'source').is_dir() and any(p.is_file() for p in (root/'source').rglob('*')), 'source empty'; "
        "assert not any(p.is_symlink() for p in root.rglob('*')), 'embedded bundle contains symlink'; "
        f"value=verify_bundle(root, expected_model_id={EXPECTED_MODEL_ID!r}, "
        f"expected_model_revision={EXPECTED_MODEL_REVISION!r}, expected_source_commit={EXPECTED_SOURCE_COMMIT!r}); "
        f"assert sha256_file(root/'bundle_manifest.json') == "
        f"{manifest['embedded_model_bundle']['bundle_manifest_sha256']!r}, 'embedded bundle manifest hash mismatch'; "
        "[importlib.import_module(name) for name in ('torch','transformers','qwen_vl_utils')]; "
        "print(json.dumps({'status':'PASS','model_id':value['model_id'],"
        "'model_revision':value['model_revision'],'source_commit':value['source_commit'],"
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
        "bundle_manifest_sha256": manifest["embedded_model_bundle"]["bundle_manifest_sha256"],
    }
    if result != expected:
        raise RuntimeError(f"embedded model verification result mismatch: expected={expected}, observed={result}")
    return result


def load_and_verify_images(bundle_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    archive = bundle_root / str(manifest["image_archive"]["path"])
    run(["docker", "load", "--input", str(archive)])
    images = verify_local_images(manifest)
    embedded = verify_embedded_model_image(manifest)
    return {"images": images, "embedded_model_bundle": embedded}


def compose_command(bundle_root: Path, env_file: Path, *arguments: str) -> list[str]:
    return [
        "docker", "compose", "--env-file", str(env_file),
        "-f", str(bundle_root / "docker-compose.offline-gpu.yml"), *arguments,
    ]


def verify_env_file(env_file: Path, *, expected_git_sha: str) -> dict[str, str]:
    if not env_file.is_file():
        raise FileNotFoundError(f"offline environment file is missing: {env_file}")
    text = env_file.read_text(encoding="utf-8")
    if "CHANGE_ME" in text:
        raise ValueError("offline environment still contains CHANGE_ME placeholder credentials")
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    if values.get("MVI_IMAGE_TAG") != expected_git_sha:
        raise ValueError(
            f"MVI_IMAGE_TAG must match bundle Git SHA {expected_git_sha}, got {values.get('MVI_IMAGE_TAG')!r}"
        )
    for key in ("POSTGRES_PASSWORD", "CLICKHOUSE_PASSWORD", "API_TOKEN", "DATA_ROOT"):
        if not values.get(key):
            raise ValueError(f"offline environment {key} must be set")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and load a registry-free MVI offline deployment bundle")
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--env-file", type=Path, help="configured .env.offline; required with --start")
    parser.add_argument("--skip-load", action="store_true", help="static verification only; no Docker runtime needed")
    parser.add_argument("--start", action="store_true", help="start the verified stack without build or pull")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    bundle_root = args.bundle_root.expanduser().resolve()
    if args.start and args.skip_load:
        parser.error("--start cannot be combined with --skip-load")
    if args.start and args.env_file is None:
        parser.error("--env-file is required with --start")

    result = verify_static_bundle(bundle_root)
    runtime: dict[str, Any] | None = None
    loaded: dict[str, Any] = {"images": [], "embedded_model_bundle": None}
    if not args.skip_load:
        runtime = verify_runtime()
        loaded = load_and_verify_images(bundle_root, result["manifest"])
    compose_config = None
    started = False
    if args.env_file is not None:
        env_file = args.env_file.expanduser().resolve()
        verify_env_file(env_file, expected_git_sha=str(result["manifest"]["git_sha"]))
        compose_config = run(compose_command(bundle_root, env_file, "config", "--format", "json"), cwd=bundle_root)
        if args.start:
            run(compose_command(bundle_root, env_file, "up", "-d", "--no-build", "--pull", "never"), cwd=bundle_root)
            started = True
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "target_platform": EXPECTED_PLATFORM,
        "verified_file_count": result["verified_file_count"],
        "runtime": runtime,
        "loaded_images": loaded["images"],
        "embedded_model_bundle": loaded["embedded_model_bundle"] or result["embedded_model_bundle"],
        "compose_config_verified": compose_config is not None,
        "stack_started": started,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
