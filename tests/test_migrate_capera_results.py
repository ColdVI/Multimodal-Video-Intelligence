import dataclasses
import json
import pathlib

import pytest

from datasets.capera import CapERAAdapter
from scripts.migrate_capera_results import build_migrated_artifact

_SOURCE = pathlib.Path("data/downloads/capera/all_results.json")
pytestmark = pytest.mark.skipif(not _SOURCE.exists(), reason="CapERA all_results.json yok")


def _real_raw_results():
    return json.loads(_SOURCE.read_text(encoding="utf-8"))


def test_migrated_artifact_preserves_real_numbers_exactly():
    raw = _real_raw_results()
    manifest = dataclasses.asdict(CapERAAdapter().manifest())
    migrated = build_migrated_artifact(raw, manifest)
    for row in raw:
        model = row["model"]
        assert model in migrated["results"]
        # her alan BIREBIR ayni - migrasyon YENIDEN HESAPLAMAZ, sadece tasir
        assert migrated["results"][model] == row


def test_migrated_artifact_has_dataset_manifest_and_known_gaps():
    raw = _real_raw_results()
    manifest = dataclasses.asdict(CapERAAdapter().manifest())
    migrated = build_migrated_artifact(raw, manifest)
    assert migrated["dataset_manifest"]["dataset_id"] == "capera"
    assert migrated["dataset_manifest"]["item_count"] == 2864
    assert len(migrated["known_gaps"]) >= 3
    assert any("ebind" in gap for gap in migrated["known_gaps"])
    assert any("2863" in gap for gap in migrated["known_gaps"])


def test_migrated_artifact_does_not_claim_clickhouse_backend():
    raw = _real_raw_results()
    manifest = dataclasses.asdict(CapERAAdapter().manifest())
    migrated = build_migrated_artifact(raw, manifest)
    assert "ClickHouse" in migrated["protocol"]
    assert "YAZILMAZ" in migrated["protocol"]


def test_on_disk_artifact_matches_what_build_function_would_produce():
    """scripts/migrate_capera_results.py --> artifacts/capera_validation.json
    zaten kosulduysa, ciktinin build_migrated_artifact() ile TUTARLI oldugunu
    dogrular (staleness kontrolu)."""
    out_path = pathlib.Path("artifacts/capera_validation.json")
    if not out_path.exists():
        pytest.skip("artifacts/capera_validation.json henuz uretilmedi")
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    raw = _real_raw_results()
    manifest = dataclasses.asdict(CapERAAdapter().manifest())
    expected = build_migrated_artifact(raw, manifest)
    assert on_disk["results"] == expected["results"]
    assert on_disk["dataset_manifest"] == expected["dataset_manifest"]
