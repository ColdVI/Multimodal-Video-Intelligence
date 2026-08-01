from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.export_offline_bundle import required_images, verify_saved_tar, write_checksums
from scripts import verify_offline_bundle as verifier
from scripts.verify_offline_bundle import validate_compose_contract, verify_checksums


ROOT = Path(__file__).resolve().parents[2]


def _manifest(tag: str = "abc123") -> dict:
    return {
        "git_sha": tag,
        "images": [{"ref": image, "image_id": f"sha256:{index}"} for index, image in enumerate(required_images(tag))],
    }


def test_offline_bundle_manifest_lists_three_images():
    manifest = _manifest()
    refs = {item["ref"] for item in manifest["images"]}
    assert refs == {
        "mvi-app-gpu:abc123",
        "pgvector/pgvector:pg16",
        "clickhouse/clickhouse-server:25.8",
    }
    assert len(refs) == 3
    compose = validate_compose_contract(ROOT / "docker-compose.offline-gpu.yml", manifest)
    assert set(compose["services"]) == {"pg", "ch", "api", "ui"}


def test_saved_tar_contains_three_expected_tags(tmp_path):
    expected = required_images("abc123")
    records = []
    archive_path = tmp_path / "images.tar"
    with tarfile.open(archive_path, "w") as archive:
        for index, tag in enumerate(expected):
            layer = f"layer-{index}/layer.tar"
            payload = b"layer"
            info = tarfile.TarInfo(layer)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            records.append({"Config": f"config-{index}.json", "RepoTags": [tag], "Layers": [layer]})
        manifest_bytes = json.dumps(records).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    result = verify_saved_tar(archive_path, expected)
    assert result["repo_tags"] == sorted(expected)
    assert result["record_count"] == 3


def test_offline_bundle_checksum_verification(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    payload = tmp_path / "nested" / "payload.bin"
    payload.write_bytes(b"offline-payload")
    entries = write_checksums(tmp_path)
    assert len(entries) == 2
    verified = verify_checksums(tmp_path)
    assert set(verified) == {"bundle_manifest.json", "nested/payload.bin"}

    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(tmp_path)


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
