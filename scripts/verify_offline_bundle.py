from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any



EXPECTED_SERVICES = {"pg", "ch", "api", "ui"}
EXPECTED_PLATFORM = "linux/amd64"


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
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_files != set(expected):
        raise ValueError(
            f"bundle inventory mismatch: missing={sorted(set(expected) - actual_files)}, "
            f"unexpected={sorted(actual_files - set(expected))}"
        )
    for relative, digest in expected.items():
        path = bundle_root / Path(relative)
        if path.is_symlink():
            raise ValueError(f"offline bundle must not contain symlinks: {relative}")
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(f"checksum mismatch for {relative}: expected {digest}, got {observed}")
    return expected


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"model bundle contains a symlink: {path}")
        if path.is_file():
            size = path.stat().st_size
            total += size
            result.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            })
    if not result:
        raise ValueError(f"model bundle subtree is empty: {root}")
    return result, total


def verify_model_bundle(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model bundle schema")
    for kind in ("source", "model"):
        detail_path = bundle_root / f"{kind}_manifest.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        files, total = _inventory(bundle_root / kind)
        if files != detail.get("files") or total != detail.get("total_size_bytes"):
            raise ValueError(f"{kind} model-bundle inventory verification failed")
        if manifest.get(f"{kind}_manifest_sha256") != sha256_file(detail_path):
            raise ValueError(f"{kind} model-bundle manifest checksum failed")
    return manifest


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
    required = {
        "ENABLED_VECTOR_BACKENDS": "clickhouse",
        "DEFAULT_VECTOR_BACKEND": "clickhouse",
        "ENABLED_DIMENSIONS": '"512"',
        "EMBEDDING_MODE": "real",
    }
    for key, expected in required.items():
        if not re.search(rf"(?m)^      {re.escape(key)}:\s*{re.escape(expected)}\s*$", api):
            raise ValueError(f"offline API {key} must be {expected!r}")
    if not re.search(r"(?m)^    gpus:\s*all\s*$", api):
        raise ValueError("offline API must request all NVIDIA GPUs")
    manifest_refs = {item["ref"] for item in manifest.get("images", [])}
    for fixed in ("pgvector/pgvector:pg16", "clickhouse/clickhouse-server:25.8"):
        if fixed not in manifest_refs:
            raise ValueError(f"bundle manifest does not list {fixed}")
    git_sha = manifest.get("git_sha")
    for application in (f"mvi-api-gpu:{git_sha}", f"mvi-ui:{git_sha}"):
        if application not in manifest_refs:
            raise ValueError(f"bundle manifest does not list {application}")
    return {"services": {name: {"image": image} for name, image in images.items()}}
def verify_static_bundle(bundle_root: Path) -> dict[str, Any]:
    bundle_root = bundle_root.expanduser().resolve()
    checksums = verify_checksums(bundle_root)
    manifest = json.loads((bundle_root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported offline bundle schema")
    if manifest.get("target_platform") != EXPECTED_PLATFORM:
        raise ValueError(f"offline target must be {EXPECTED_PLATFORM}")
    compose = validate_compose_contract(bundle_root / "docker-compose.offline-gpu.yml", manifest)
    requirement = manifest.get("requirements") or {}
    requirement_path = bundle_root / str(requirement.get("path", "requirements.txt"))
    if sha256_file(requirement_path) != requirement.get("sha256"):
        raise ValueError("canonical requirements.txt hash does not match bundle manifest")
    model = verify_model_bundle(bundle_root / str(manifest["model_bundle"]["path"]))
    archive = bundle_root / str(manifest["image_archive"]["path"])
    if archive.stat().st_size != manifest["image_archive"].get("size_bytes"):
        raise ValueError("image archive size does not match bundle manifest")
    if sha256_file(archive) != manifest["image_archive"].get("sha256"):
        raise ValueError("image archive hash does not match bundle manifest")
    return {
        "manifest": manifest,
        "compose": compose,
        "model_manifest": model,
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


def load_and_verify_images(bundle_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    archive = bundle_root / str(manifest["image_archive"]["path"])
    run(["docker", "load", "--input", str(archive)])
    result: list[dict[str, Any]] = []
    for image in manifest["images"]:
        payload = json.loads(run(["docker", "image", "inspect", image["ref"]]))[0]
        observed = f"{str(payload.get('Os', '')).lower()}/{str(payload.get('Architecture', '')).lower()}"
        if observed != EXPECTED_PLATFORM:
            raise RuntimeError(f"loaded image {image['ref']} has wrong platform: {observed}")
        if payload.get("Id") != image.get("image_id"):
            raise RuntimeError(f"loaded image ID mismatch for {image['ref']}")
        result.append({"ref": image["ref"], "image_id": payload.get("Id"), "platform": observed})
    return result


def compose_command(bundle_root: Path, env_file: Path, *arguments: str) -> list[str]:
    return [
        "docker", "compose", "--env-file", str(env_file),
        "-f", str(bundle_root / "docker-compose.offline-gpu.yml"), *arguments,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and load a registry-free MVI offline deployment bundle")
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--env-file", type=Path, help="configured .env.offline; required with --start")
    parser.add_argument("--skip-load", action="store_true", help="verify only; do not docker load the image archive")
    parser.add_argument("--start", action="store_true", help="start the verified stack without build or pull")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    bundle_root = args.bundle_root.expanduser().resolve()
    if args.start and args.skip_load:
        parser.error("--start cannot be combined with --skip-load")
    if args.start and args.env_file is None:
        parser.error("--env-file is required with --start")

    result = verify_static_bundle(bundle_root)
    runtime = verify_runtime()
    loaded = [] if args.skip_load else load_and_verify_images(bundle_root, result["manifest"])
    compose_config = None
    started = False
    if args.env_file is not None:
        env_file = args.env_file.expanduser().resolve()
        compose_config = run(compose_command(bundle_root, env_file, "config", "--format", "json"), cwd=bundle_root)
        if args.start:
            run(compose_command(bundle_root, env_file, "up", "-d", "--no-build", "--pull", "never"), cwd=bundle_root)
            started = True
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "target_platform": EXPECTED_PLATFORM,
        "verified_file_count": result["verified_file_count"],
        "runtime": runtime,
        "loaded_images": loaded,
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
