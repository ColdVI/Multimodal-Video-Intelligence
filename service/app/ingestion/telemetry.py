from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from app.ingestion.manifest import ABSOLUTE_CLOCKS, DatasetManifest


@dataclass(frozen=True)
class TelemetryRecord:
    timestamp: float | str
    values: Mapping[str, Any]


class TelemetryAdapter(Protocol):
    def iter_records(self, source: Path) -> Iterator[TelemetryRecord]:
        ...


def _unix_seconds(timestamp: float | str, clock: str) -> float:
    if clock == "unix_ms":
        return float(timestamp) / 1000.0
    if clock == "unix_s":
        return float(timestamp)
    if clock == "iso8601":
        text = str(timestamp).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    raise ValueError(f"clock {clock!r} is not absolute")


def align_timestamp(
    timestamp: float | str,
    *,
    telemetry_clock: str,
    offset_s: float,
    video_start_unix_s: float | None,
) -> float:
    """Align telemetry to video PTS.

    Absolute clocks use ``telemetry_unix_s - video_start_unix_s - offset_s``.
    Relative clocks use ``telemetry_relative_s - offset_s``. Therefore a
    positive offset moves telemetry earlier on the video timeline.
    """
    if telemetry_clock == "relative_s":
        return float(timestamp) - offset_s
    if telemetry_clock in ABSOLUTE_CLOCKS:
        if video_start_unix_s is None:
            raise ValueError("absolute telemetry clock requires video_start_unix_s")
        return _unix_seconds(timestamp, telemetry_clock) - video_start_unix_s - offset_s
    raise ValueError(f"unsupported telemetry clock: {telemetry_clock!r}")


class GenericCSVAdapter:
    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest

    def iter_records(self, source: Path) -> Iterator[TelemetryRecord]:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or self.manifest.timestamp_column not in reader.fieldnames:
                raise ValueError(
                    f"telemetry CSV {source} lacks timestamp column {self.manifest.timestamp_column!r}"
                )
            required_sources = {
                field.source for field in (
                    *self.manifest.telemetry_fields.values(),
                    *self.manifest.telemetry_extra.values(),
                )
            }
            missing = sorted(required_sources - set(reader.fieldnames))
            if missing:
                raise ValueError(f"telemetry CSV {source} lacks mapped columns: {missing}")
            for row in reader:
                yield TelemetryRecord(row[self.manifest.timestamp_column], row)


def canonical_values(record: TelemetryRecord, manifest: DatasetManifest) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, field in manifest.telemetry_fields.items():
        raw = record.values.get(field.source)
        if raw in (None, ""):
            result[name] = None
            continue
        if field.data_type == "categorical":
            result[name] = str(raw)
        elif field.data_type == "boolean":
            result[name] = str(raw).strip().lower() in {"1", "true", "yes", "on"}
        elif field.data_type == "integer":
            result[name] = int(float(raw) * field.scale + field.offset)
        else:
            result[name] = float(raw) * field.scale + field.offset
    return result


__all__ = [
    "GenericCSVAdapter", "TelemetryAdapter", "TelemetryRecord", "align_timestamp",
    "canonical_values",
]
