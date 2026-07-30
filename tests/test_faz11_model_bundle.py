from __future__ import annotations

import subprocess

import pytest

from scripts.prepare_model_bundle import create_bundle
from app.embedding.bundle import verify_bundle


def _git(command, cwd):
    result = subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


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
