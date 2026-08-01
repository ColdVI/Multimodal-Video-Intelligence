from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_offline_bundle import required_images, write_checksums
from scripts.verify_offline_bundle import validate_compose_contract, verify_checksums


ROOT = Path(__file__).resolve().parents[2]


def _manifest(tag: str = "abc123") -> dict:
    return {
        "git_sha": tag,
        "images": [{"ref": image} for image in required_images(tag)],
    }


def test_offline_bundle_manifest_lists_all_images():
    manifest = _manifest()
    refs = {item["ref"] for item in manifest["images"]}
    assert refs == {
        "mvi-api-gpu:abc123",
        "mvi-ui:abc123",
        "pgvector/pgvector:pg16",
        "clickhouse/clickhouse-server:25.8",
    }
    compose = validate_compose_contract(ROOT / "docker-compose.offline-gpu.yml", manifest)
    assert set(compose["services"]) == {"pg", "ch", "api", "ui"}


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
