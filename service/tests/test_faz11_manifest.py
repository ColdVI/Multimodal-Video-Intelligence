from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.ingestion.manifest import derive_identifier, load_manifest
from app.ingestion.telemetry import align_timestamp
from app.preflight import discover_pairs, run_data_preflight


def _payload(*, clock="relative_s"):
    alignment = {
        "video_clock": "pts", "telemetry_clock": clock, "offset_s": 1.5,
        "max_gap_s": 1.0, "missing_policy": None,
    }
    if clock != "relative_s":
        alignment["video_start_time_from"] = "container_creation_time"
    return {
        "schema_version": 1,
        "dataset_id": "fixture",
        "display_name": "Fixture",
        "source": {"videos_glob": "videos/*.mp4", "video_id_from": "filename_stem"},
        "pairing": {
            "strategy": "filename_stem", "telemetry_glob": "telemetry/*.csv",
            "telemetry_id_from": "filename_stem", "manifest_csv": None,
        },
        "time_alignment": alignment,
        "window": {
            "size_s": 8.0, "stride_s": 4.0, "frames_per_item": 8,
            "partial_window_policy": "drop_partial",
        },
        "telemetry": {
            "format": "generic_csv", "timestamp_column": "timestamp",
            "fields": {
                "altitude_m": {
                    "source": "alt", "unit": "m", "reference": "AGL", "type": "continuous",
                },
                "velocity_mps": {
                    "source": "speed", "unit": "m/s", "kind": "ground_speed", "type": "continuous",
                },
                "compass_heading": {"source": "heading", "unit": "deg", "type": "circular_deg"},
            },
            "extra": {"battery_v": {"source": "battery", "unit": "V", "type": "continuous"}},
        },
        "media": {"enabled": True, "clip_cache": True},
    }


def _manifest(tmp_path: Path, payload=None):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload or _payload(), sort_keys=False), encoding="utf-8")
    return load_manifest(path)


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env({
        "ARTIFACTS_ROOT": str(tmp_path),
        "ENABLED_VECTOR_BACKENDS": "clickhouse",
        "ENABLED_DIMENSIONS": "512",
    })


def test_repository_example_manifest_is_valid():
    manifest = load_manifest(Path(__file__).resolve().parents[2] / "datasets" / "example_uav.yaml")
    assert manifest.dataset_id == "kurum_ucuslari"
    assert manifest.telemetry_fields["altitude_m"].reference == "AGL"
    assert manifest.telemetry_fields["velocity_mps"].kind == "ground_speed"
    assert manifest.telemetry_extra["sensor_mode"].data_type == "categorical"


def test_absolute_glob_is_rejected(tmp_path):
    payload = _payload()
    payload["source"]["videos_glob"] = "/mnt/nas/*.mp4"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="relative to DATA_ROOT"):
        load_manifest(path)


def test_parent_traversal_is_rejected(tmp_path):
    payload = _payload()
    payload["pairing"]["telemetry_glob"] = "../outside/*.csv"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain"):
        load_manifest(path)


def test_altitude_and_velocity_semantics_are_required(tmp_path):
    payload = _payload()
    del payload["telemetry"]["fields"]["altitude_m"]["reference"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="altitude_m requires reference"):
        load_manifest(path)


def test_circular_fields_cannot_use_linear_interpolation(tmp_path):
    payload = _payload()
    payload["telemetry"]["fields"]["compass_heading"]["interpolation"] = "linear"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires circular interpolation"):
        load_manifest(path)


def test_relative_clock_does_not_require_anchor():
    assert align_timestamp(12.0, telemetry_clock="relative_s", offset_s=2.0, video_start_unix_s=None) == 10.0


def test_absolute_clock_requires_and_uses_anchor():
    with pytest.raises(ValueError, match="requires video_start_unix_s"):
        align_timestamp(11_000, telemetry_clock="unix_ms", offset_s=1.0, video_start_unix_s=None)
    assert align_timestamp(11_000, telemetry_clock="unix_ms", offset_s=1.0, video_start_unix_s=5.0) == 5.0


def test_categorical_field_is_not_linearly_interpolated(tmp_path):
    payload = _payload()
    payload["telemetry"]["extra"]["sensor_mode"] = {"source": "mode", "type": "categorical", "interpolation": "linear"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="categorical field .* requires locf interpolation and mode aggregation"):
        load_manifest(path)


def test_categorical_field_defaults_to_locf_and_mode(tmp_path):
    payload = _payload()
    payload["telemetry"]["extra"]["sensor_mode"] = {"source": "mode", "type": "categorical"}
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    manifest = load_manifest(path)
    field = manifest.telemetry_extra["sensor_mode"]
    assert field.interpolation == "locf" and field.aggregation == "mode"


def test_timezone_naive_absolute_timestamp_uses_manifest_timezone_not_hardcoded_utc():
    """Regression test: telemetry.py::_unix_seconds used to hardcode UTC for a
    naive iso8601 timestamp regardless of the manifest's declared timezone,
    unlike preflight.py/generic_loader.py which both already localized the
    video-start anchor via ZoneInfo(manifest.timezone). A naive timestamp must
    be interpreted according to the manifest's explicit setting, not silently
    forced to UTC when the operator declared something else."""
    naive = "2026-01-01T12:00:00"
    utc_result = align_timestamp(
        naive, telemetry_clock="iso8601", offset_s=0.0, video_start_unix_s=0.0, timezone_name="UTC",
    )
    istanbul_result = align_timestamp(
        naive, telemetry_clock="iso8601", offset_s=0.0, video_start_unix_s=0.0, timezone_name="Europe/Istanbul",
    )
    # Istanbul (UTC+3, no DST in January) is 3 hours ahead of UTC, so the same
    # naive wall-clock reading corresponds to an earlier absolute instant.
    assert istanbul_result == pytest.approx(utc_result - 3 * 3600.0, abs=1.0)
    # An explicit UTC offset in the timestamp itself must win regardless of
    # the manifest's configured timezone (it is no longer naive).
    explicit_utc = align_timestamp(
        "2026-01-01T12:00:00+00:00", telemetry_clock="iso8601", offset_s=0.0,
        video_start_unix_s=0.0, timezone_name="Europe/Istanbul",
    )
    assert explicit_utc == pytest.approx(utc_result, abs=1e-6)


def test_offset_sign_matches_documentation():
    """align_timestamp's docstring states a positive offset moves telemetry
    earlier on the video timeline, for both relative and absolute clocks."""
    baseline_relative = align_timestamp(10.0, telemetry_clock="relative_s", offset_s=0.0, video_start_unix_s=None)
    offset_relative = align_timestamp(10.0, telemetry_clock="relative_s", offset_s=2.0, video_start_unix_s=None)
    assert offset_relative < baseline_relative
    assert offset_relative == pytest.approx(baseline_relative - 2.0)

    baseline_absolute = align_timestamp(1000.0, telemetry_clock="unix_s", offset_s=0.0, video_start_unix_s=0.0)
    offset_absolute = align_timestamp(1000.0, telemetry_clock="unix_s", offset_s=5.0, video_start_unix_s=0.0)
    assert offset_absolute < baseline_absolute
    assert offset_absolute == pytest.approx(baseline_absolute - 5.0)


def test_heading_fields_are_never_implicitly_cross_mapped(tmp_path):
    """The manifest schema has no auto-mapping logic: yaw, compass_heading, and
    gimbal_heading are only ever populated when the operator explicitly maps a
    source column to that exact canonical name. Declaring only compass_heading
    must never cause yaw to be silently populated from the same source."""
    payload = _payload()
    assert "yaw" not in payload["telemetry"]["fields"]
    assert "compass_heading" in payload["telemetry"]["fields"]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    manifest = load_manifest(path)
    assert "compass_heading" in manifest.telemetry_fields
    assert "yaw" not in manifest.telemetry_fields


def test_altitude_datum_must_be_one_of_the_explicit_reference_values(tmp_path):
    payload = _payload()
    payload["telemetry"]["fields"]["altitude_m"]["reference"] = "ellipsoid_typo"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="altitude_m requires reference"):
        load_manifest(path)


def test_velocity_semantics_are_preserved_on_the_parsed_field(tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest.telemetry_fields["velocity_mps"].kind == "ground_speed"
    payload = _payload()
    payload["telemetry"]["fields"]["velocity_mps"]["kind"] = "air_speed"
    path = tmp_path / "air.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_manifest(path).telemetry_fields["velocity_mps"].kind == "air_speed"


def test_extra_telemetry_is_not_dropped_alongside_canonical_fields(tmp_path):
    manifest = _manifest(tmp_path)
    assert "battery_v" in manifest.telemetry_extra
    assert "battery_v" not in manifest.telemetry_fields
    from app.ingestion.telemetry import AlignedTelemetryRecord, TelemetrySeries

    series = TelemetrySeries([
        AlignedTelemetryRecord(0.0, {"altitude_m": 10.0, "compass_heading": 10.0}, {"battery_v": 12.0}),
        AlignedTelemetryRecord(4.0, {"altitude_m": 12.0, "compass_heading": 20.0}, {"battery_v": 11.5}),
    ], manifest)
    canonical, extra = series.aggregate_window(0.0, 4.0)
    assert canonical["altitude_m"] is not None
    assert extra["battery_v"] is not None
    assert "battery_v" not in canonical
    assert "altitude_m" not in extra


def test_identifier_regex_requires_named_video_id():
    assert derive_identifier(Path("flight_part2.mp4"), r"regex:^(?P<video_id>.+?)_part\d+$") == "flight"
    with pytest.raises(ValueError, match="named video_id"):
        derive_identifier(Path("flight.mp4"), r"regex:^(.*)$")


@dataclass(frozen=True)
class _Probe:
    duration_s: float = 16.0
    fps: float = 25.0
    width: int = 640
    height: int = 480
    creation_time: datetime | None = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_preflight_pairs_files_and_estimates_segments(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "videos").mkdir()
    (tmp_path / "telemetry").mkdir()
    (tmp_path / "videos" / "flight.mp4").write_bytes(b"fixture-not-decoded")
    (tmp_path / "telemetry" / "flight.csv").write_text(
        "timestamp,alt,speed,heading,battery\n0,10,2,359,12\n4,11,3,1,11.9\n",
        encoding="utf-8",
    )
    pairs = discover_pairs(manifest, tmp_path)
    assert len(pairs) == 1 and pairs[0].video_id == "flight"
    report = run_data_preflight(
        manifest.path, data_root=tmp_path, configured=_settings(tmp_path), probe_fn=lambda _: _Probe(),
    )
    assert report["status"] == "not_run"  # Model bundle check belongs to Aşama 4.
    assert report["video_count"] == 1
    assert report["estimated_segments"] == 3
    assert not [item for item in report["checks"] if item["status"] == "fail"]


def test_duplicate_video_ids_fail_preflight(tmp_path):
    payload = _payload()
    payload["source"]["video_id_from"] = r"regex:^(?P<video_id>same)-\d+$"
    manifest = _manifest(tmp_path, payload)
    (tmp_path / "videos").mkdir()
    (tmp_path / "telemetry").mkdir()
    for name in ("same-1", "same-2"):
        (tmp_path / "videos" / f"{name}.mp4").write_bytes(b"x")
        (tmp_path / "telemetry" / f"{name}.csv").write_text(
            "timestamp,alt,speed,heading,battery\n0,10,2,0,12\n", encoding="utf-8",
        )
    report = run_data_preflight(
        manifest.path, data_root=tmp_path, configured=_settings(tmp_path), probe_fn=lambda _: _Probe(),
    )
    check = next(item for item in report["checks"] if item["check_id"] == "duplicate_video_ids")
    assert check["status"] == "fail"
