from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from app.embedding.bundle import inventory, sha256_file as model_sha256, verify_bundle
from scripts import export_offline_bundle as exporter
from scripts import verify_offline_bundle as verifier


ROOT = Path(__file__).resolve().parents[2]


def _docker_archive(
    path: Path,
    refs: tuple[str, ...],
    *,
    architecture_by_ref: dict[str, str] | None = None,
    omit_last_layer: bool = False,
) -> dict[str, str]:
    records = []
    image_ids: dict[str, str] = {}
    with tarfile.open(path, "w") as archive:
        for index, ref in enumerate(refs):
            architecture = (architecture_by_ref or {}).get(ref, "amd64")
            config_bytes = json.dumps({"architecture": architecture, "os": "linux", "index": index}).encode()
            config_name = f"config-{index}.json"
            config_info = tarfile.TarInfo(config_name)
            config_info.size = len(config_bytes)
            archive.addfile(config_info, io.BytesIO(config_bytes))
            image_ids[ref] = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
            layer = f"layer-{index}/layer.tar"
            if not (omit_last_layer and index == len(refs) - 1):
                payload = b"layer"
                layer_info = tarfile.TarInfo(layer)
                layer_info.size = len(payload)
                archive.addfile(layer_info, io.BytesIO(payload))
            records.append({"Config": config_name, "RepoTags": [ref], "Layers": [layer]})
        manifest_bytes = json.dumps(records).encode()
        manifest_info = tarfile.TarInfo("manifest.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
    return image_ids


def _manifest(tag: str = "abc123", image_ids: dict[str, str] | None = None) -> dict:
    refs = exporter.required_images(tag)
    ids = image_ids or {ref: f"sha256:{index}" for index, ref in enumerate(refs)}
    return {
        "schema_version": 2,
        "git_sha": tag,
        "target_platform": "linux/amd64",
        "images": [
            {"ref": ref, "image_id": ids[ref], "os": "linux", "architecture": "amd64"}
            for ref in refs
        ],
        "embedded_model_bundle": {
            "image_ref": f"mvi-app-gpu:{tag}",
            "container_root": "/opt/mvi-model-bundle",
            "model_id": exporter.EXPECTED_MODEL_ID,
            "model_revision": exporter.EXPECTED_MODEL_REVISION,
            "source_commit": exporter.EXPECTED_SOURCE_COMMIT,
            "bundle_manifest_sha256": "a" * 64,
        },
        "embedded_model_container_smoke": {"status": "PASS"},
    }


def _static_bundle(tmp_path: Path, *, architecture_by_ref: dict[str, str] | None = None) -> Path:
    root = tmp_path / "bundle"
    (root / "images").mkdir(parents=True)
    (root / "datasets").mkdir()
    (root / "scripts").mkdir()
    for relative in (
        "docker-compose.offline-gpu.yml",
        ".env.offline.example",
        "datasets/video_only_m2ts.yaml",
        "scripts/verify_offline_bundle.py",
        "install-and-start-offline.sh",
    ):
        target = root / relative
        target.write_bytes((ROOT / relative).read_bytes())
    tar_path = root / "images" / "mvi-images-linux-amd64.tar"
    refs = exporter.required_images("abc123")
    image_ids = _docker_archive(tar_path, refs, architecture_by_ref=architecture_by_ref)
    manifest = _manifest(image_ids=image_ids)
    manifest["image_archive"] = {
        "path": "images/mvi-images-linux-amd64.tar",
        "size_bytes": tar_path.stat().st_size,
        "sha256": exporter.sha256_file(tar_path),
        "repo_tags": sorted(refs),
        "record_count": 3,
    }
    manifest["actual_sizes"] = {
        "model_bundle_external_transfer_size_bytes": 0,
        "transfer_payload_size_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "bundle_directory_size_bytes": 0,
    }
    for _ in range(5):
        (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        exporter.write_checksums(root)
        observed = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if manifest["actual_sizes"]["bundle_directory_size_bytes"] == observed:
            break
        manifest["actual_sizes"]["bundle_directory_size_bytes"] = observed
    else:
        raise AssertionError("test fixture size did not stabilize")
    return root


def _model_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "model-bundle"
    (root / "model").mkdir(parents=True)
    (root / "source").mkdir()
    (root / "model" / "weights.safetensors").write_bytes(b"weights")
    (root / "source" / "modeling.py").write_text("VALUE = 1\n", encoding="utf-8")
    details = {}
    for kind in ("source", "model"):
        files, total = inventory(root / kind)
        detail = {"kind": kind, "files": files, "total_size_bytes": total}
        path = root / f"{kind}_manifest.json"
        path.write_text(json.dumps(detail, sort_keys=True), encoding="utf-8")
        details[f"{kind}_manifest_sha256"] = model_sha256(path)
    manifest = {
        "schema_version": 1,
        "model_id": exporter.EXPECTED_MODEL_ID,
        "model_revision": exporter.EXPECTED_MODEL_REVISION,
        "source_commit": exporter.EXPECTED_SOURCE_COMMIT,
        **details,
    }
    (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_offline_bundle_manifest_lists_exactly_three_images():
    manifest = _manifest()
    refs = {item["ref"] for item in manifest["images"]}
    assert refs == {
        "mvi-app-gpu:abc123",
        "pgvector/pgvector:pg16",
        "clickhouse/clickhouse-server:25.8",
    }
    assert len(refs) == 3
    compose = verifier.validate_compose_contract(ROOT / "docker-compose.offline-gpu.yml", manifest)
    assert set(compose["services"]) == {"pg", "ch", "api", "ui"}


def test_saved_tar_contains_three_expected_tags_and_amd64_configs(tmp_path):
    expected = exporter.required_images("abc123")
    archive_path = tmp_path / "images.tar"
    _docker_archive(archive_path, expected)
    result = exporter.verify_saved_tar(archive_path, expected)
    assert result["repo_tags"] == sorted(expected)
    assert result["record_count"] == 3
    assert {item["platform"] for item in result["images"].values()} == {"linux/amd64"}


def test_saved_tar_rejects_missing_tag_and_layer(tmp_path):
    expected = exporter.required_images("abc123")
    missing_tag = tmp_path / "missing-tag.tar"
    _docker_archive(missing_tag, expected[:-1])
    with pytest.raises(ValueError, match="tag set mismatch"):
        exporter.verify_saved_tar(missing_tag, expected)

    missing_layer = tmp_path / "missing-layer.tar"
    _docker_archive(missing_layer, expected, omit_last_layer=True)
    with pytest.raises(ValueError, match="missing config/layer"):
        exporter.verify_saved_tar(missing_layer, expected)


def test_arm64_application_archive_is_rejected(tmp_path):
    expected = exporter.required_images("abc123")
    archive = tmp_path / "arm64.tar"
    _docker_archive(archive, expected, architecture_by_ref={expected[0]: "arm64"})
    with pytest.raises(ValueError, match="platform mismatch"):
        exporter.verify_saved_tar(archive, expected)


def test_offline_bundle_checksum_verification(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    payload = tmp_path / "nested" / "payload.bin"
    payload.write_bytes(b"offline-payload")
    entries = exporter.write_checksums(tmp_path)
    assert len(entries) == 2
    assert set(verifier.verify_checksums(tmp_path)) == {"bundle_manifest.json", "nested/payload.bin"}
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier.verify_checksums(tmp_path)


def test_checksum_writer_rejects_symlinked_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "payload").write_text("payload", encoding="utf-8")
    link = tmp_path / "linked-directory"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        exporter.write_checksums(tmp_path)


def test_small_fixture_static_bundle_verifies_without_external_model_directory(tmp_path):
    root = _static_bundle(tmp_path)
    result = verifier.verify_static_bundle(root)
    assert result["manifest"]["schema_version"] == 2
    assert result["embedded_model_bundle"]["container_root"] == "/opt/mvi-model-bundle"
    assert not (root / "model-bundle").exists()
    assert result["verified_file_count"] == 7


def test_static_verifier_rejects_old_schema(tmp_path):
    root = _static_bundle(tmp_path)
    path = root / "bundle_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    exporter.write_checksums(root)
    with pytest.raises(ValueError, match="expected schema_version 2"):
        verifier.verify_static_bundle(root)


def test_wrong_model_revision_and_source_commit_are_rejected(tmp_path):
    root = _model_bundle(tmp_path)
    with pytest.raises(ValueError, match="model_revision mismatch"):
        verify_bundle(root, expected_model_revision="wrong-revision")
    with pytest.raises(ValueError, match="source_commit mismatch"):
        verify_bundle(root, expected_source_commit="wrong-commit")


def test_model_bundle_symlink_is_rejected(tmp_path):
    root = _model_bundle(tmp_path)
    link = root / "model" / "linked.bin"
    try:
        link.symlink_to(root / "model" / "weights.safetensors")
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        verify_bundle(root)


def test_compose_model_mount_regression_is_rejected(tmp_path):
    compose = tmp_path / "compose.yml"
    text = (ROOT / "docker-compose.offline-gpu.yml").read_text(encoding="utf-8")
    text = text.replace(
        "      - ./datasets:/workspace/datasets:ro",
        "      - ./datasets:/workspace/datasets:ro\n"
        "      - ${MODEL_BUNDLE_ROOT:-./model-bundle}:/opt/mvi-model-bundle:ro",
    )
    compose.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must not mount"):
        verifier.validate_compose_contract(compose, _manifest())


def test_export_contract_copies_no_model_or_video_dataset(tmp_path):
    output = tmp_path / "contract"
    output.mkdir()
    exporter._copy_contract_files(output, "abc123")
    assert not (output / "model-bundle").exists()
    assert not (output / "data").exists()
    assert not (output / "videos").exists()
    assert "MODEL_BUNDLE_ROOT=" not in (output / ".env.offline.example").read_text(encoding="utf-8")
    assert "mvi-app-gpu:abc123" not in (output / "docker-compose.offline-gpu.yml").read_text(encoding="utf-8")
    env_text = (output / ".env.offline.example").read_text(encoding="utf-8")
    assert env_text.startswith("# Copy")
    assert "MVI_IMAGE_TAG=abc123" in env_text


def test_exporter_embedded_model_metadata_contains_pins_and_manifest_hash(tmp_path):
    root = _model_bundle(tmp_path)
    verified = verify_bundle(root)
    metadata = exporter.embedded_model_metadata("mvi-app-gpu:abc123", verified, root)
    assert metadata == {
        "image_ref": "mvi-app-gpu:abc123",
        "container_root": "/opt/mvi-model-bundle",
        "model_id": exporter.EXPECTED_MODEL_ID,
        "model_revision": exporter.EXPECTED_MODEL_REVISION,
        "source_commit": exporter.EXPECTED_SOURCE_COMMIT,
        "bundle_manifest_sha256": model_sha256(root / "bundle_manifest.json"),
    }


def test_prepare_verify_only_and_export_estimate_need_no_site_packages(tmp_path):
    root = _model_bundle(tmp_path)
    verify = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / "prepare_model_bundle.py"),
         "--bundle-root", str(root), "--verify-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["model_revision"] == exporter.EXPECTED_MODEL_REVISION
    estimate = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / "export_offline_bundle.py"),
         "--output-dir", str(tmp_path / "unused"), "--model-bundle", str(root), "--estimate-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert estimate.returncode == 0, estimate.stderr
    assert json.loads(estimate.stdout)["target_platform"] == "linux/amd64"


def test_embedded_model_runtime_smoke_disables_network(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **_):
        calls.append(command)
        return json.dumps({
            "status": "PASS",
            "model_id": exporter.EXPECTED_MODEL_ID,
            "model_revision": exporter.EXPECTED_MODEL_REVISION,
            "source_commit": exporter.EXPECTED_SOURCE_COMMIT,
            "bundle_manifest_sha256": "a" * 64,
        })

    monkeypatch.setattr(verifier, "run", fake_run)
    assert verifier.verify_embedded_model_image(_manifest())["status"] == "PASS"
    assert calls[0][:7] == ["docker", "run", "--rm", "--pull", "never", "--network", "none"]
    assert "mvi-app-gpu:abc123" in calls[0]


def test_missing_image_fails_without_pull(monkeypatch):
    calls: list[list[str]] = []

    def missing(command, **_):
        calls.append(command)
        raise RuntimeError("No such image")

    monkeypatch.setattr(verifier, "run", missing)
    with pytest.raises(RuntimeError, match="refusing to pull"):
        verifier.verify_local_images(_manifest())
    assert calls == [["docker", "image", "inspect", "mvi-app-gpu:abc123"]]
    assert all("pull" not in command for command in calls)


def test_offline_environment_rejects_placeholders_and_wrong_image_tag(tmp_path):
    env_file = tmp_path / ".env.offline"
    env_file.write_text("MVI_IMAGE_TAG=abc123\nPOSTGRES_PASSWORD=CHANGE_ME\n", encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        verifier.verify_env_file(env_file, expected_git_sha="abc123")
    env_file.write_text(
        "MVI_IMAGE_TAG=wrong\nPOSTGRES_PASSWORD=secret\nCLICKHOUSE_PASSWORD=secret\n"
        "API_TOKEN=secret\nDATA_ROOT=/videos\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must match bundle Git SHA"):
        verifier.verify_env_file(env_file, expected_git_sha="abc123")


def test_builder_and_starter_shell_contracts():
    builder = (ROOT / "scripts" / "build_offline_bundle_macos.sh").read_text(encoding="utf-8")
    starter = (ROOT / "install-and-start-offline.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"' in builder
    assert "--target\", \"gpu-bundled" not in builder
    assert "--target gpu-bundled" not in builder  # exporter owns the one canonical build command
    assert "--target\", \"gpu-bundled" in Path(exporter.__file__).read_text(encoding="utf-8")
    assert "FROM gpu AS gpu-bundled" in dockerfile
    assert "COPY --from=mvi_model_bundle / /opt/mvi-model-bundle/" in dockerfile
    assert dockerfile.index("ARG CUDA_IMAGE_TAG=") < dockerfile.index("FROM ${PYTHON_IMAGE}")
    assert "--pull\", \"never" in Path(verifier.__file__).read_text(encoding="utf-8")
    assert "--load-only" in starter
    assert "CHANGE_ME" in starter


def test_buildx_command_uses_verified_named_context_and_amd64(monkeypatch, tmp_path):
    calls: list[tuple[list[str], Path | None]] = []

    def fake_run(command, *, cwd=None):
        calls.append((command, cwd))
        return ""

    monkeypatch.setattr(exporter, "run", fake_run)
    exporter.build_bundled_application_image("mvi-app-gpu:abc123", tmp_path, "linux/amd64")
    assert calls == [([
        "docker", "buildx", "build", "--platform", "linux/amd64",
        "--build-context", f"mvi_model_bundle={tmp_path}",
        "--target", "gpu-bundled", "--load", "-t", "mvi-app-gpu:abc123", ".",
    ], exporter.REPO_ROOT)]


def test_load_only_mode_never_requests_stack_start(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    starter = bundle / "install-and-start-offline.sh"
    starter.write_bytes((ROOT / "install-and-start-offline.sh").read_bytes())
    starter.chmod(0o755)
    (bundle / ".env.offline").write_text(
        "MVI_IMAGE_TAG=abc123\nPOSTGRES_PASSWORD=secret\nCLICKHOUSE_PASSWORD=secret\nAPI_TOKEN=secret\n",
        encoding="utf-8",
    )
    calls = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS_FILE"\n', encoding="utf-8")
    python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CALLS_FILE"] = str(calls)
    result = subprocess.run(["bash", str(starter), "--load-only"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invoked = calls.read_text(encoding="utf-8")
    assert "--start" not in invoked
    assert "--env-file" in invoked
