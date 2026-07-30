from __future__ import annotations

import csv
import bisect
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from app.ingestion.manifest import ABSOLUTE_CLOCKS, DatasetManifest


@dataclass(frozen=True)
class TelemetryRecord:
    timestamp: float | str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class AlignedTelemetryRecord:
    time_s: float
    canonical: Mapping[str, Any]
    extra: Mapping[str, Any]


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
    return _mapped_values(record, manifest.telemetry_fields)


def extra_values(record: TelemetryRecord, manifest: DatasetManifest) -> dict[str, Any]:
    return _mapped_values(record, manifest.telemetry_extra)


def _mapped_values(record: TelemetryRecord, fields: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, field in fields.items():
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


def circular_interpolate(left: float, right: float, fraction: float) -> float:
    delta = (right - left + 180.0) % 360.0 - 180.0
    return (left + fraction * delta) % 360.0


def circular_mean(values: list[float]) -> float | None:
    if not values:
        return None
    sin_sum = sum(math.sin(math.radians(value)) for value in values)
    cos_sum = sum(math.cos(math.radians(value)) for value in values)
    if math.isclose(sin_sum, 0.0, abs_tol=1e-12) and math.isclose(cos_sum, 0.0, abs_tol=1e-12):
        return None
    result = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    return 0.0 if math.isclose(result, 360.0, abs_tol=1e-12) else result


class TelemetrySeries:
    def __init__(self, records: list[AlignedTelemetryRecord], manifest: DatasetManifest):
        if any(left.time_s > right.time_s for left, right in zip(records, records[1:])):
            raise ValueError("telemetry timestamps must be monotonic")
        self.records = records
        self.times = [item.time_s for item in self.records]
        self.manifest = manifest

    @classmethod
    def from_csv(
        cls,
        path: Path,
        manifest: DatasetManifest,
        *,
        video_start_unix_s: float | None,
        offset_s: float | None = None,
    ) -> "TelemetrySeries":
        rows = []
        for record in GenericCSVAdapter(manifest).iter_records(path):
            rows.append(AlignedTelemetryRecord(
                time_s=align_timestamp(
                    record.timestamp,
                    telemetry_clock=manifest.telemetry_clock,
                    offset_s=manifest.time_offset_s if offset_s is None else offset_s,
                    video_start_unix_s=video_start_unix_s,
                ),
                canonical=canonical_values(record, manifest),
                extra=extra_values(record, manifest),
            ))
        return cls(rows, manifest)

    def _interpolate(self, field_name: str, at_s: float, *, extra: bool = False) -> Any:
        fields = self.manifest.telemetry_extra if extra else self.manifest.telemetry_fields
        field = fields[field_name]
        position = bisect.bisect_left(self.times, at_s)
        if position < len(self.times) and math.isclose(self.times[position], at_s, abs_tol=1e-9):
            values = self.records[position].extra if extra else self.records[position].canonical
            return values.get(field_name)
        before = position - 1
        after = position
        if field.data_type == "categorical":
            if before < 0 or at_s - self.times[before] > self.manifest.max_gap_s:
                return None
            return (self.records[before].extra if extra else self.records[before].canonical).get(field_name)
        if before < 0 or after >= len(self.records):
            return None
        left_gap = at_s - self.times[before]
        right_gap = self.times[after] - at_s
        if max(left_gap, right_gap) > self.manifest.max_gap_s:
            return None
        left_values = self.records[before].extra if extra else self.records[before].canonical
        right_values = self.records[after].extra if extra else self.records[after].canonical
        left = left_values.get(field_name)
        right = right_values.get(field_name)
        if left is None or right is None:
            return None
        fraction = left_gap / (left_gap + right_gap)
        if field.data_type == "circular_deg":
            return circular_interpolate(float(left), float(right), fraction)
        if field.data_type == "boolean":
            return bool(left if fraction < 0.5 else right)
        return float(left) + (float(right) - float(left)) * fraction

    def _aggregate_field(self, name: str, start_s: float, end_s: float, *, extra: bool = False) -> Any:
        fields = self.manifest.telemetry_extra if extra else self.manifest.telemetry_fields
        field = fields[name]
        start_index = bisect.bisect_left(self.times, start_s)
        end_index = bisect.bisect_right(self.times, end_s)
        values = []
        for record in self.records[start_index:end_index]:
            source = record.extra if extra else record.canonical
            value = source.get(name)
            if value is not None:
                values.append(value)
        for boundary in (start_s, end_s):
            value = self._interpolate(name, boundary, extra=extra)
            if value is not None:
                values.append(value)
        if not values:
            return None
        if field.aggregation == "circular_mean":
            return circular_mean([float(value) for value in values])
        if field.aggregation == "mode":
            modes = statistics.multimode(values)
            return modes[0] if modes else None
        if field.data_type == "boolean":
            return sum(bool(value) for value in values) >= len(values) / 2
        value = statistics.median(float(item) for item in values)
        return int(round(value)) if field.data_type == "integer" else value

    def aggregate_window(self, start_s: float, end_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
        canonical = {
            name: self._aggregate_field(name, start_s, end_s)
            for name in self.manifest.telemetry_fields
        }
        extra = {
            name: self._aggregate_field(name, start_s, end_s, extra=True)
            for name in self.manifest.telemetry_extra
        }
        return canonical, extra


__all__ = [
    "AlignedTelemetryRecord", "GenericCSVAdapter", "TelemetryAdapter", "TelemetryRecord",
    "TelemetrySeries", "align_timestamp", "canonical_values", "circular_interpolate",
    "circular_mean", "extra_values",
]
