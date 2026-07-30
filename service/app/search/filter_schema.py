from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

from app.ingestion.manifest import DatasetManifest


FilterType = Literal["float", "integer", "keyword", "boolean", "circular_deg"]


@dataclass(frozen=True)
class FilterField:
    name: str
    display_name: str
    source: str
    data_type: FilterType
    unit: str | None
    filterable: bool
    indexed: bool


CANONICAL_FILTER_FIELDS: dict[str, FilterField] = {
    "event_category": FilterField("event_category", "Event category", "video", "keyword", None, True, True),
    "split": FilterField("split", "Split", "video", "keyword", None, True, True),
    "video_id": FilterField("video_id", "Video", "segment", "keyword", None, True, True),
    "latitude": FilterField("latitude", "Latitude", "telemetry", "float", "deg", True, False),
    "longitude": FilterField("longitude", "Longitude", "telemetry", "float", "deg", True, False),
    "altitude_m": FilterField("altitude_m", "Altitude", "telemetry", "float", "m", True, True),
    "velocity_mps": FilterField("velocity_mps", "Velocity", "telemetry", "float", "m/s", True, True),
    "roll": FilterField("roll", "Roll", "telemetry", "float", "deg", True, False),
    "pitch": FilterField("pitch", "Pitch", "telemetry", "float", "deg", True, False),
    "yaw": FilterField("yaw", "Yaw", "telemetry", "circular_deg", "deg", True, False),
    "yaw_rate": FilterField("yaw_rate", "Yaw rate", "telemetry", "float", "deg/s", True, False),
    "gimbal_pitch": FilterField("gimbal_pitch", "Gimbal pitch", "telemetry", "float", "deg", True, True),
    "gimbal_heading": FilterField("gimbal_heading", "Gimbal heading", "telemetry", "circular_deg", "deg", True, False),
    "compass_heading": FilterField("compass_heading", "Compass heading", "telemetry", "circular_deg", "deg", True, False),
    "person_count": FilterField("person_count", "People", "metadata", "integer", None, True, True),
    "vehicle_count": FilterField("vehicle_count", "Vehicles", "metadata", "integer", None, True, True),
    "bus_count": FilterField("bus_count", "Buses", "metadata", "integer", None, True, True),
    "is_night": FilterField("is_night", "Night", "metadata", "boolean", None, True, True),
}


def manifest_filter_fields(manifest: DatasetManifest) -> tuple[FilterField, ...]:
    fields = []
    for name, definition in manifest.telemetry_fields.items():
        base = CANONICAL_FILTER_FIELDS[name]
        data_type: FilterType = (
            "float" if definition.data_type == "continuous" else
            "integer" if definition.data_type == "integer" else
            "boolean" if definition.data_type == "boolean" else
            "circular_deg" if definition.data_type == "circular_deg" else "keyword"
        )
        fields.append(FilterField(
            name, base.display_name, base.source, data_type, definition.unit, True, base.indexed,
        ))
    for name in ("event_category", "split", "video_id"):
        fields.append(CANONICAL_FILTER_FIELDS[name])
    return tuple(dict.fromkeys(fields))


def serialize_fields(fields: Mapping[str, FilterField] | tuple[FilterField, ...]) -> list[dict]:
    values = fields.values() if isinstance(fields, Mapping) else fields
    return [asdict(field) for field in values]


__all__ = ["CANONICAL_FILTER_FIELDS", "FilterField", "manifest_filter_fields", "serialize_fields"]
