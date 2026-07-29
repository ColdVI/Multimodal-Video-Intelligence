import dataclasses
import json
import pathlib

import pytest

from datasets.capera import CapERAAdapter
from scripts.migrate_capera_results import build_migrated_artifact, evaluated_counts

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


def test_manifest_counts_and_evaluated_counts_are_kept_separate_with_real_data():
    """istegin ozu: 2864/14320 (manifest) TAMAMI degerlendirildi VARSAYILMAZ -
    all_results.json GERCEKTEN 2863/14315/0-basarisiz raporluyor, ikisi
    farkli alanlarda durur."""
    raw = _real_raw_results()
    manifest = dataclasses.asdict(CapERAAdapter().manifest())
    migrated = build_migrated_artifact(raw, manifest)
    counts = migrated["counts"]
    assert counts["manifest_video_count"] == 2864
    assert counts["manifest_query_count"] == 2864 * 5
    assert counts["evaluated_video_count"] == 2863  # all_results.json'dan GERCEK
    assert counts["evaluated_query_count"] == 14315  # all_results.json'dan GERCEK
    assert counts["failed_video_count"] == 0
    # manifest ile evaluated FARKLI - biri digerinin tamami degil
    assert counts["manifest_video_count"] != counts["evaluated_video_count"]


def test_evaluated_counts_is_unknown_when_models_disagree():
    # gercekci olmayan ama olasi durum: iki model FARKLI n_videos raporlarsa
    # hangisi 'dogru' bilinemez - uydurmak yerine unknown.
    fake_results = [
        {"model": "A", "n_videos": 100, "n_queries": 500, "n_failed_videos": 0},
        {"model": "B", "n_videos": 99, "n_queries": 500, "n_failed_videos": 0},
    ]
    counts = evaluated_counts(fake_results)
    assert counts["evaluated_video_count"] == "unknown"
    assert counts["evaluated_query_count"] == 500  # bu alanda anlasiyorlar, gercek deger kalir
    assert counts["failed_video_count"] == 0


def test_evaluated_counts_is_unknown_when_field_missing():
    fake_results = [{"model": "A", "n_queries": 500}]  # n_videos, n_failed_videos yok
    counts = evaluated_counts(fake_results)
    assert counts["evaluated_video_count"] == "unknown"
    assert counts["evaluated_query_count"] == 500
    assert counts["failed_video_count"] == "unknown"


def test_evaluated_counts_is_unknown_when_no_results_at_all():
    counts = evaluated_counts([])
    assert counts["evaluated_video_count"] == "unknown"
    assert counts["evaluated_query_count"] == "unknown"
    assert counts["failed_video_count"] == "unknown"


def test_build_migrated_artifact_never_assumes_full_manifest_evaluated_when_unknown():
    manifest = {"item_count": 999, "query_count": 4995}
    migrated = build_migrated_artifact([], manifest)
    counts = migrated["counts"]
    assert counts["manifest_video_count"] == 999
    assert counts["evaluated_video_count"] == "unknown"
    assert counts["evaluated_video_count"] != counts["manifest_video_count"]


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
    assert on_disk["counts"] == expected["counts"]
