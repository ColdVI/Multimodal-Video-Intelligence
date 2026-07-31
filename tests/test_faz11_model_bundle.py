from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.prepare_model_bundle import create_bundle
from app.embedding.bundle import verify_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(command, cwd):
    result = subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _fixture_bundle(tmp_path, *, model_id="org/model", model_revision="revision"):
    source = tmp_path / "source-checkout"
    source.mkdir()
    _git(["init"], source)
    _git(["config", "user.email", "fixture@example.invalid"], source)
    _git(["config", "user.name", "Fixture"], source)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "module.py"], source)
    _git(["commit", "-m", "fixture"], source)
    commit = _git(["rev-parse", "HEAD"], source)
    model = tmp_path / "model-snapshot"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    bundle = tmp_path / "bundle"
    create_bundle(
        bundle, model_id=model_id, model_revision=model_revision,
        source_repo="https://example.invalid/source.git", source_commit=commit,
        source_path=source, model_path=model,
    )
    return bundle, commit


def test_local_bundle_is_pinned_hashed_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source-checkout"
    source.mkdir()
    _git(["init"], source)
    _git(["config", "user.email", "fixture@example.invalid"], source)
    _git(["config", "user.name", "Fixture"], source)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "module.py"], source)
    _git(["commit", "-m", "fixture"], source)
    commit = _git(["rev-parse", "HEAD"], source)
    model = tmp_path / "model-snapshot"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    bundle = tmp_path / "bundle"

    manifest = create_bundle(
        bundle, model_id="org/model", model_revision="model-revision",
        source_repo="https://example.invalid/source.git", source_commit=commit,
        source_path=source, model_path=model,
    )

    assert manifest["source_commit"] == commit
    assert manifest["model_revision"] == "model-revision"
    assert verify_bundle(
        bundle, expected_model_id="org/model", expected_model_revision="model-revision",
        expected_source_commit=commit,
    )["total_size_bytes"] > 0
    assert not (bundle / "source" / ".git").exists()
    with pytest.raises(FileExistsError):
        create_bundle(
            bundle, model_id="org/model", model_revision="model-revision",
            source_repo="https://example.invalid/source.git", source_commit=commit,
            source_path=source, model_path=model,
        )


def test_bundle_verification_detects_modified_file(tmp_path):
    source = tmp_path / "source-checkout"
    source.mkdir()
    _git(["init"], source)
    _git(["config", "user.email", "fixture@example.invalid"], source)
    _git(["config", "user.name", "Fixture"], source)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "module.py"], source)
    _git(["commit", "-m", "fixture"], source)
    commit = _git(["rev-parse", "HEAD"], source)
    model = tmp_path / "model-snapshot"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    bundle = tmp_path / "bundle"
    create_bundle(
        bundle, model_id="org/model", model_revision="revision", source_repo="source",
        source_commit=commit, source_path=source, model_path=model,
    )
    (bundle / "model" / "weights.bin").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash/size"):
        verify_bundle(bundle)


def test_bundle_accepts_valid_hash_chain(tmp_path):
    bundle, commit = _fixture_bundle(tmp_path)
    manifest = verify_bundle(
        bundle, expected_model_id="org/model", expected_model_revision="revision",
        expected_source_commit=commit,
    )
    assert manifest["source_commit"] == commit
    assert manifest["model_revision"] == "revision"


def test_bundle_rejects_modified_source_file(tmp_path):
    bundle, _ = _fixture_bundle(tmp_path)
    (bundle / "source" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size"):
        verify_bundle(bundle)


def test_bundle_rejects_missing_model_file(tmp_path):
    bundle, _ = _fixture_bundle(tmp_path)
    (bundle / "model" / "weights.bin").unlink()
    with pytest.raises(ValueError, match="hash/size"):
        verify_bundle(bundle)


def test_bundle_rejects_revision_mismatch(tmp_path):
    bundle, commit = _fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="model_revision mismatch"):
        verify_bundle(bundle, expected_model_revision="some-other-revision")
    with pytest.raises(ValueError, match="source_commit mismatch"):
        verify_bundle(bundle, expected_source_commit="0" * 40)
    with pytest.raises(ValueError, match="model_id mismatch"):
        verify_bundle(bundle, expected_model_id="org/different-model")


def test_bundle_rejects_tampered_detail_manifest(tmp_path):
    """Tampering with source_manifest.json itself (not a source file) must be
    caught via the bundle_manifest.json-recorded manifest hash, independent of
    the file-inventory comparison covered by test_bundle_rejects_modified_source_file."""
    bundle, _ = _fixture_bundle(tmp_path)
    detail_path = bundle / "source_manifest.json"
    detail_path.write_text(detail_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash verification failed"):
        verify_bundle(bundle)


def test_compose_mounts_bundle_read_only():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8"))
    volumes = compose["services"]["api"]["volumes"]
    assert volumes, "docker-compose.gpu.yml must mount the model bundle"
    for volume in volumes:
        assert volume.rstrip().endswith(":ro"), f"model bundle mount must be read-only: {volume}"
    # host QWEN_REPO_PATH/QWEN_MODEL_PATH must match the container targets Compose mounts
    environment = compose["services"]["api"]["environment"]
    assert environment["QWEN_REPO_PATH"] in " ".join(volumes)
    assert environment["QWEN_MODEL_PATH"] in " ".join(volumes)


def test_dockerfile_gpu_never_clones_or_downloads_at_build_time():
    dockerfile = (REPO_ROOT / "service" / "Dockerfile.gpu").read_text(encoding="utf-8")
    forbidden = ("git clone", "git checkout", "snapshot_download", "huggingface-cli download", "wget ", "curl ")
    for token in forbidden:
        assert token not in dockerfile, f"Dockerfile.gpu must not fetch model/source at build time: {token!r}"
    assert "CUDA_IMAGE_TAG" in dockerfile
