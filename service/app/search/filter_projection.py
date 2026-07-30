from __future__ import annotations


POSTGRES_RUN_COLUMNS = {
    "event_category": "v.event_category", "split": "v.split", "video_id": "s.video_id",
    "latitude": "t.latitude", "longitude": "t.longitude", "altitude_m": "t.altitude_m",
    "velocity_mps": "t.velocity_mps", "roll": "t.roll", "pitch": "t.pitch", "yaw": "t.yaw",
    "yaw_rate": "t.yaw_rate", "gimbal_pitch": "t.gimbal_pitch",
    "gimbal_heading": "t.gimbal_heading", "compass_heading": "t.compass_heading",
    "person_count": "m.person_count", "vehicle_count": "m.vehicle_count",
    "bus_count": "m.bus_count", "is_night": "(coalesce(t.extra->>'is_night','false'))::boolean",
}

CLICKHOUSE_COLUMNS = {name: name for name in POSTGRES_RUN_COLUMNS}


__all__ = ["CLICKHOUSE_COLUMNS", "POSTGRES_RUN_COLUMNS"]
