from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = 1
ABSOLUTE_CLOCKS = {"unix_ms", "unix_s", "iso8601"}
RELATIVE_CLOCKS = {"relative_s"}
CANONICAL_FIELDS = (
    "event_category", "split", "video_id", "latitude", "longitude",
    "altitude_m", "velocity_mps", "roll", "pitch", "yaw", "yaw_rate",
    "gimbal_pitch", "gimbal_heading", "compass_heading", "person_count",
    "vehicle_count", "bus_count", "is_night",
)
CIRCULAR_FIELDS = {"yaw", "gimbal_heading", "compass_heading"}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def validate_relative_path(value: str | None, location: str) -> str | None:
    if value in (None, ""):
        return None
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or re.match(r"^[A-Za-z]:/", str(value)):
        raise ValueError(f"{location} must be relative to DATA_ROOT")
    if ".." in path.parts:
        raise ValueError(f"{location} must not contain '..'")
    return str(value)


@dataclass(frozen=True)
class TelemetryField:
    source: str
    data_type: str
    unit: str | None = None
    reference: str | None = None
    kind: str | None = None
    interpolation: str | None = None
    aggregation: str | None = None
    scale: float = 1.0
    offset: float = 0.0

    @classmethod
    def parse(cls, name: str, value: Any, *, canonical: bool) -> "TelemetryField":
        raw = _mapping(value, f"telemetry.{'fields' if canonical else 'extra'}.{name}")
        source = str(raw.get("source", "")).strip()
        if not source:
            raise ValueError(f"telemetry field {name!r} requires source")
        data_type = str(raw.get("type", "continuous"))
        if data_type not in {"continuous", "integer", "categorical", "boolean", "circular_deg"}:
            raise ValueError(f"telemetry field {name!r} has unsupported type {data_type!r}")
        interpolation = raw.get("interpolation")
        aggregation = raw.get("aggregation")
        if data_type == "circular_deg":
            if interpolation not in (None, "circular") or aggregation not in (None, "circular_mean"):
                raise ValueError(f"circular field {name!r} requires circular interpolation and circular_mean aggregation")
            interpolation = "circular"
            aggregation = "circular_mean"
        elif data_type == "categorical":
            interpolation = str(interpolation or "locf")
            aggregation = str(aggregation or "mode")
        else:
            interpolation = str(interpolation or "linear")
            aggregation = str(aggregation or "median")
        if name == "altitude_m" and canonical and raw.get("reference") not in {"AGL", "MSL", "WGS84"}:
            raise ValueError("altitude_m requires reference: AGL, MSL, or WGS84")
        if name == "velocity_mps" and canonical and raw.get("kind") not in {"ground_speed", "air_speed"}:
            raise ValueError("velocity_mps requires kind: ground_speed or air_speed")
        if name in CIRCULAR_FIELDS and canonical and data_type != "circular_deg":
            raise ValueError(f"canonical heading field {name!r} must use type: circular_deg")
        return cls(
            source=source,
            data_type=data_type,
            unit=None if raw.get("unit") is None else str(raw["unit"]),
            reference=None if raw.get("reference") is None else str(raw["reference"]),
            kind=None if raw.get("kind") is None else str(raw["kind"]),
            interpolation=interpolation,
            aggregation=aggregation,
            scale=float(raw.get("scale", 1.0)),
            offset=float(raw.get("offset", 0.0)),
        )


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    schema_version: int
    dataset_id: str
    display_name: str
    videos_glob: str
    video_id_from: str
    pairing_strategy: str
    telemetry_glob: str | None
    telemetry_id_from: str
    pairing_manifest_csv: str | None
    video_clock: str
    telemetry_clock: str
    video_start_time_from: str | None
    filename_time_regex: str | None
    filename_time_format: str | None
    timezone: str
    time_offset_s: float
    max_gap_s: float
    missing_policy: str | None
    window_size_s: float
    stride_s: float
    frames_per_item: int
    partial_window_policy: str
    telemetry_format: str
    timestamp_column: str
    telemetry_fields: Mapping[str, TelemetryField]
    telemetry_extra: Mapping[str, TelemetryField]
    media_enabled: bool
    media_clip_cache: bool
    fail_on_video_error: bool
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def is_absolute_clock(self) -> bool:
        return self.telemetry_clock in ABSOLUTE_CLOCKS

    def resolve_glob(self, data_root: Path, pattern: str) -> tuple[Path, ...]:
        validate_relative_path(pattern, "manifest glob")
        root = data_root.expanduser().resolve()
        matches: list[Path] = []
        for path in root.glob(pattern):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"resolved source escapes DATA_ROOT: {path}")
            if path.is_file():
                matches.append(path)
        return tuple(sorted(matches))


def _parse_manifest(path: Path, raw: Mapping[str, Any]) -> DatasetManifest:
    schema_version = int(raw.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    dataset_id = str(raw.get("dataset_id", "")).strip()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", dataset_id):
        raise ValueError("dataset_id must contain only letters, numbers, dot, dash, and underscore")
    source = _mapping(raw.get("source"), "source")
    pairing = _mapping(raw.get("pairing", {}), "pairing")
    alignment = _mapping(raw.get("time_alignment"), "time_alignment")
    window = _mapping(raw.get("window"), "window")
    telemetry = _mapping(raw.get("telemetry"), "telemetry")
    fields_raw = _mapping(telemetry.get("fields", {}), "telemetry.fields")
    extra_raw = _mapping(telemetry.get("extra", {}), "telemetry.extra")
    unknown_fields = sorted(set(fields_raw) - set(CANONICAL_FIELDS))
    if unknown_fields:
        raise ValueError(f"non-canonical telemetry fields belong under telemetry.extra: {unknown_fields}")
    telemetry_clock = str(alignment.get("telemetry_clock", ""))
    if telemetry_clock not in ABSOLUTE_CLOCKS | RELATIVE_CLOCKS:
        raise ValueError(f"unsupported telemetry_clock: {telemetry_clock!r}")
    size_s = float(window.get("size_s", 0))
    stride_s = float(window.get("stride_s", 0))
    frames_per_item = int(window.get("frames_per_item", 0))
    if size_s <= 0 or stride_s <= 0 or frames_per_item <= 0:
        raise ValueError("window size_s, stride_s, and frames_per_item must be positive")
    videos_glob = validate_relative_path(str(source.get("videos_glob", "")), "source.videos_glob")
    if not videos_glob:
        raise ValueError("source.videos_glob is required")
    if float(alignment.get("max_gap_s", 1.0)) <= 0:
        raise ValueError("time_alignment.max_gap_s must be positive")
    partial = str(window.get("partial_window_policy", "drop_partial"))
    if partial not in {"drop_partial", "pad_last"}:
        raise ValueError("partial_window_policy must be drop_partial or pad_last")
    strategy = str(pairing.get("strategy", "filename_stem"))
    if strategy not in {"filename_stem", "manifest_csv"}:
        raise ValueError("pairing.strategy must be filename_stem or manifest_csv")
    manifest_csv = validate_relative_path(pairing.get("manifest_csv"), "pairing.manifest_csv")
    if strategy == "manifest_csv" and manifest_csv is None:
        raise ValueError("pairing.manifest_csv is required for manifest_csv strategy")
    video_start_from = alignment.get("video_start_time_from")
    if telemetry_clock in ABSOLUTE_CLOCKS and video_start_from not in {
        "container_creation_time", "filename", "manifest_csv"
    }:
        raise ValueError("absolute telemetry clock requires a supported video_start_time_from anchor")
    media = _mapping(raw.get("media", {}), "media")
    policy = _mapping(raw.get("policy", {}), "policy")
    fields = {str(name): TelemetryField.parse(str(name), value, canonical=True) for name, value in fields_raw.items()}
    extra = {str(name): TelemetryField.parse(str(name), value, canonical=False) for name, value in extra_raw.items()}
    return DatasetManifest(
        path=path,
        schema_version=schema_version,
        dataset_id=dataset_id,
        display_name=str(raw.get("display_name") or dataset_id),
        videos_glob=videos_glob,
        video_id_from=str(source.get("video_id_from", "filename_stem")),
        pairing_strategy=strategy,
        telemetry_glob=validate_relative_path(pairing.get("telemetry_glob"), "pairing.telemetry_glob"),
        telemetry_id_from=str(pairing.get("telemetry_id_from", "filename_stem")),
        pairing_manifest_csv=manifest_csv,
        video_clock=str(alignment.get("video_clock", "pts")),
        telemetry_clock=telemetry_clock,
        video_start_time_from=None if video_start_from is None else str(video_start_from),
        filename_time_regex=None if alignment.get("filename_time_regex") is None else str(alignment["filename_time_regex"]),
        filename_time_format=None if alignment.get("filename_time_format") is None else str(alignment["filename_time_format"]),
        timezone=str(alignment.get("timezone", "UTC")),
        time_offset_s=float(alignment.get("offset_s", 0.0)),
        max_gap_s=float(alignment.get("max_gap_s", 1.0)),
        missing_policy=None if alignment.get("missing_policy") is None else str(alignment["missing_policy"]),
        window_size_s=size_s,
        stride_s=stride_s,
        frames_per_item=frames_per_item,
        partial_window_policy=partial,
        telemetry_format=str(telemetry.get("format", "generic_csv")),
        timestamp_column=str(telemetry.get("timestamp_column", "timestamp")),
        telemetry_fields=fields,
        telemetry_extra=extra,
        media_enabled=bool(media.get("enabled", True)),
        media_clip_cache=bool(media.get("clip_cache", True)),
        fail_on_video_error=bool(policy.get("fail_on_video_error", True)),
        raw=raw,
    )


def load_manifest(path: str | Path) -> DatasetManifest:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return _parse_manifest(target.resolve(), _mapping(raw, "manifest"))


def derive_identifier(path: Path, rule: str) -> str:
    if rule == "filename_stem":
        return path.stem
    if rule.startswith("regex:"):
        match = re.match(rule.removeprefix("regex:"), path.stem)
        if match is None or "video_id" not in match.groupdict():
            raise ValueError(f"video_id_from regex did not produce named video_id for {path.name}")
        return str(match.group("video_id"))
    raise ValueError(f"unsupported identifier rule: {rule!r}")


__all__ = [
    "ABSOLUTE_CLOCKS", "CANONICAL_FIELDS", "DatasetManifest", "RELATIVE_CLOCKS",
    "SCHEMA_VERSION", "TelemetryField", "derive_identifier", "load_manifest",
    "validate_relative_path",
]
