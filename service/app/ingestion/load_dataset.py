from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DIMENSIONS, settings
from app.db import clickhouse, postgres, qdrant
from app.embedding.router import embed_item
from app.mrl import truncate_and_normalize


@dataclass
class DatasetBundle:
    dataset: tuple[Any, ...]
    videos: list[tuple[Any, ...]]
    segments: list[tuple[Any, ...]]
    metadata: list[tuple[Any, ...]]
    telemetry: list[tuple[Any, ...]]
    vector_payload: list[dict[str, Any]]
    quantile_frame: pd.DataFrame
    groundtruth: list[tuple[Any, ...]] = field(default_factory=list)


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _none(value: Any) -> Any:
    if value is None:
        return None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _float(value: Any) -> float | None:
    value = _none(value)
    return None if value is None else float(value)


def _int(value: Any, default: int = 0) -> int:
    value = _none(value)
    return default if value is None else int(value)


def load_auair_bundle() -> DatasetBundle:
    root = settings.artifacts_root / "research"
    segment_path = root / "auair_segments.parquet"
    telemetry_path = root / "auair_telemetry.parquet"
    segments_df = pd.read_parquet(segment_path)
    telemetry_df = pd.read_parquet(telemetry_path)
    frame = segments_df.merge(telemetry_df, on="segment_id", how="left", validate="one_to_one")
    if frame["segment_id"].duplicated().any() or len(frame) != 1866:
        raise ValueError("AU-AIR contract failed: expected 1866 unique segments")
    altitude_max = float(frame["altitude_m"].max())
    if altitude_max > 1000:
        frame["altitude_m"] = frame["altitude_m"] / 1000.0
        altitude_max = float(frame["altitude_m"].max())
    if not 1.0 <= altitude_max <= 100.0:
        raise ValueError(f"AU-AIR altitude unit/range check failed: max={altitude_max}")

    video_rows: list[tuple[Any, ...]] = []
    for video_id, group in frame.groupby("video_id", sort=True):
        video_rows.append(
            ("auair", str(video_id), f"/workspace/data/research/auair/images.zip::{video_id}", "train", float(group["t_end"].max()), None)
        )

    segment_rows: list[tuple[Any, ...]] = []
    metadata_rows: list[tuple[Any, ...]] = []
    telemetry_rows: list[tuple[Any, ...]] = []
    payload_rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        segment_id = str(record["segment_id"])
        person_count = _int(record.get("person_count"))
        vehicle_count = _int(record.get("vehicle_count"))
        object_classes = []
        if person_count:
            object_classes.append("person")
        if vehicle_count:
            object_classes.append("vehicle")
        segment_rows.append(
            (segment_id, "auair", str(record["video_id"]), float(record["t_start"]), float(record["t_end"]), None)
        )
        metadata_rows.append((segment_id, person_count, vehicle_count, 0, object_classes, None, None))
        telemetry_rows.append(
            (
                segment_id, None, None, None, None,
                _float(record.get("altitude_m")), _float(record.get("velocity_mps")),
                _float(record.get("roll")), _float(record.get("pitch")), _float(record.get("yaw")),
                _float(record.get("yaw_rate")), None, None, None, None,
            )
        )
        payload_rows.append(
            {
                "segment_id": segment_id,
                "dataset_id": "auair",
                "video_id": str(record["video_id"]),
                "t_start": float(record["t_start"]),
                "t_end": float(record["t_end"]),
                "altitude_m": _float(record.get("altitude_m")),
                "velocity_mps": _float(record.get("velocity_mps")),
                "gimbal_pitch": None,
                "person_count": person_count,
                "vehicle_count": vehicle_count,
                "is_night": 0,
            }
        )
    return DatasetBundle(
        dataset=("auair", "research-parquet-v1", _sha256([segment_path, telemetry_path]), "CC BY-NC-SA", True, False),
        videos=video_rows,
        segments=segment_rows,
        metadata=metadata_rows,
        telemetry=telemetry_rows,
        vector_payload=payload_rows,
        quantile_frame=frame,
    )


def _find_json(dataset_id: str) -> Path:
    candidates = list((settings.data_root / "research" / dataset_id).glob("**/*.json"))
    if not candidates:
        raise FileNotFoundError(f"no annotation JSON found under {settings.data_root / 'research' / dataset_id}")
    return max(candidates, key=lambda value: value.stat().st_size)


def load_seadronessee_bundle() -> DatasetBundle:
    path = _find_json("seadronessee")
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", payload if isinstance(payload, list) else [])
    errors_path = settings.artifacts_root / "research" / "ingest_errors.jsonl"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    valid: list[dict[str, Any]] = []
    with errors_path.open("a", encoding="utf-8") as errors:
        for item in images:
            try:
                source = item.get("source") or {}
                meta = item.get("meta") or {}
                video_id = str(item.get("video_id", source.get("video")))
                frame_index = int(item.get("frame_index", source.get("frame_no")))
                if video_id in {"None", ""}:
                    raise ValueError("missing video_id")
                valid.append({**item, "_video_id": video_id, "_frame_index": frame_index, "_meta": meta, "_source": source})
            except Exception as exc:
                errors.write(json.dumps({"dataset_id": "seadronessee", "error": str(exc), "row": item}, ensure_ascii=False) + "\n")
    if not valid:
        raise ValueError("SeaDronesSee annotation contained no valid records")

    frame = pd.DataFrame(
        {
            "altitude_m": [_float(row["_meta"].get("altitude")) for row in valid],
            "velocity_mps": [_float(row["_meta"].get("speed")) for row in valid],
            "gimbal_pitch": [_float(row["_meta"].get("gimbal_pitch")) for row in valid],
            "person_count": [0 for _ in valid],
        }
    )
    video_ids = sorted({row["_video_id"] for row in valid})
    videos = [("seadronessee", value, next((str(r["_source"].get("video")) for r in valid if r["_video_id"] == value), value), "train", None, None) for value in video_ids]
    segments: list[tuple[Any, ...]] = []
    metadata: list[tuple[Any, ...]] = []
    telemetry: list[tuple[Any, ...]] = []
    rows: list[dict[str, Any]] = []
    for item in valid:
        start = float(item["_frame_index"])
        end = start + 1.0
        segment_id = f"seadronessee:{item['_video_id']}:{start:.3f}:{end:.3f}"
        segments.append((segment_id, "seadronessee", item["_video_id"], start, end, None))
        metadata.append((segment_id, 0, 0, 0, [], None, None))
        meta = item["_meta"]
        telemetry.append((
            segment_id, _none(item.get("date_time")), _none(item.get("date_time")),
            _float(meta.get("gps_latitude")), _float(meta.get("gps_longitude")),
            _float(meta.get("altitude")), _float(meta.get("speed")), None, None, None, None,
            _float(meta.get("gimbal_pitch")), _float(meta.get("gimbal_heading")),
            _float(meta.get("compass_heading")), None,
        ))
        rows.append({
            "segment_id": segment_id, "dataset_id": "seadronessee", "video_id": item["_video_id"],
            "t_start": start, "t_end": end, "altitude_m": _float(meta.get("altitude")),
            "velocity_mps": _float(meta.get("speed")), "gimbal_pitch": _float(meta.get("gimbal_pitch")),
            "person_count": 0, "vehicle_count": 0, "is_night": 0,
        })
    return DatasetBundle(
        dataset=("seadronessee", "annotation-json", _sha256([path]), "research use; verify upstream terms", True, False),
        videos=videos, segments=segments, metadata=metadata, telemetry=telemetry,
        vector_payload=rows, quantile_frame=frame,
    )


def load_capera_bundle() -> DatasetBundle:
    # ERA_caption semasini burada yeniden implemente etmiyoruz. Tek sozlesme
    # dataset_adapters.capera.CapERAAdapter'dir.
    from dataset_adapters.capera import CapERAAdapter

    adapter = CapERAAdapter()
    capera_cfg = adapter.cfg["datasets"]["capera"]
    split = str(capera_cfg["quality_split"])
    expected_items = int(capera_cfg["quality_item_count"])
    expected_captions = int(capera_cfg["captions_per_item"])
    expected_queries = int(capera_cfg["quality_query_count"])
    duration_s = float(capera_cfg["fixed_duration_s"])
    sequence_ids = adapter.list_sequences(split=split)
    if len(sequence_ids) != expected_items:
        raise ValueError(
            f"CapERA {split} contract failed: expected {expected_items} items, got {len(sequence_ids)}"
        )
    manifest = adapter.manifest(split=split)
    videos: list[tuple[Any, ...]] = []
    segments: list[tuple[Any, ...]] = []
    metadata: list[tuple[Any, ...]] = []
    rows: list[dict[str, Any]] = []
    groundtruth: list[tuple[Any, ...]] = []
    for video_id in sequence_ids:
        entry = adapter.entry(video_id)
        captions = adapter.captions(video_id)
        if len(captions) != expected_captions:
            raise ValueError(
                f"CapERA caption contract failed for {video_id}: "
                f"expected {expected_captions}, got {len(captions)}"
            )
        source = str(adapter.load_video(video_id))
        segment_id = f"capera:{video_id}:0.000:{duration_s:.3f}"
        videos.append(("capera", video_id, source, split, duration_s, entry["category"]))
        segments.append((segment_id, "capera", video_id, 0.0, duration_s, captions[0]))
        metadata.append((segment_id, 0, 0, 0, [], None, None))
        rows.append({
            "segment_id": segment_id, "dataset_id": "capera", "video_id": video_id,
            "t_start": 0.0, "t_end": duration_s, "altitude_m": None, "velocity_mps": None,
            "gimbal_pitch": None, "person_count": 0, "vehicle_count": 0, "is_night": 0,
        })
        for caption_index, query_text in enumerate(captions):
            query_id = f"capera:{video_id}:caption:{caption_index}"
            groundtruth.append(
                ("capera", query_id, query_text, segment_id, video_id, 1, caption_index, "unknown")
            )
    if len(groundtruth) != expected_queries or len({row[1] for row in groundtruth}) != expected_queries:
        raise ValueError(
            f"CapERA GT contract failed: expected {expected_queries} unique rows, got {len(groundtruth)}"
        )
    return DatasetBundle(
        dataset=(
            "capera", f"{manifest.dataset_version} [{split}]", manifest.source_hash,
            adapter.license(), False, True,
        ),
        videos=videos, segments=segments, metadata=metadata, telemetry=[], vector_payload=rows,
        quantile_frame=pd.DataFrame(), groundtruth=groundtruth,
    )


LOADERS = {"auair": load_auair_bundle, "capera": load_capera_bundle, "seadronessee": load_seadronessee_bundle}


def _build_vectors(bundle: DatasetBundle) -> dict[int, list[dict[str, Any]]]:
    result = {dimension: [] for dimension in DIMENSIONS}
    for payload in bundle.vector_payload:
        base = embed_item(payload["segment_id"], 2048, dataset_id=payload["dataset_id"])
        for dimension in DIMENSIONS:
            vector = base if dimension == 2048 else truncate_and_normalize(base, dimension)
            if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
                raise AssertionError(f"invalid {dimension}d embedding for {payload['segment_id']}")
            result[dimension].append({**payload, "embedding": vector.astype(np.float32).tolist()})
    return result


def _clickhouse_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = {"altitude_m", "velocity_mps", "gimbal_pitch"}
    return [
        {key: (float("nan") if key in numeric and value is None else value) for key, value in row.items()}
        for row in rows
    ]


def _write_quantiles(dataset_id: str, frame: pd.DataFrame) -> None:
    path = settings.artifacts_root / "research" / "selectivity_thresholds_v2.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    dataset_values: dict[str, Any] = {}
    for column in ("altitude_m", "velocity_mps", "gimbal_pitch", "person_count"):
        if column not in frame or frame[column].dropna().empty:
            dataset_values[column] = None
            continue
        series = frame[column].dropna().astype(float)
        dataset_values[column] = {
            "min": float(series.min()), "max": float(series.max()),
            "upper_50pct": float(series.quantile(0.50)),
            "upper_10pct": float(series.quantile(0.90)),
            "upper_1pct": float(series.quantile(0.99)),
            "upper_0_1pct": float(series.quantile(0.999)),
        }
    existing[dataset_id] = dataset_values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, Any]]) -> None:
    path = settings.artifacts_root / "research" / "ingest_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset_id", "dimension", "backend", "row_count", "elapsed_s", "storage_mb", "embedding_mode"]
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fields:
                replaced_datasets = {str(row["dataset_id"]) for row in rows}
                existing = [row for row in reader if row.get("dataset_id") not in replaced_datasets]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + rows)


def ingest(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in LOADERS:
        raise ValueError(f"unsupported dataset: {dataset_id}")
    postgres.init_schema()
    clickhouse.init_schema()
    qdrant.init_schema()
    bundle = LOADERS[dataset_id]()
    started = time.perf_counter()
    postgres.upsert_dataset_bundle(
        bundle.dataset, bundle.videos, bundle.segments, bundle.metadata,
        bundle.telemetry, bundle.groundtruth,
    )
    vectors = _build_vectors(bundle)
    report: list[dict[str, Any]] = []
    for dimension, rows in vectors.items():
        backend_started = time.perf_counter()
        triples = [(row["segment_id"], dataset_id, row["embedding"]) for row in rows]
        postgres.upsert_vectors(dimension, triples)
        if dimension == 1024:
            postgres.upsert_vectors(dimension, triples, half_1024=True)
        report.append({
            "dataset_id": dataset_id, "dimension": dimension, "backend": "pgvector",
            "row_count": len(rows), "elapsed_s": round(time.perf_counter() - backend_started, 6),
            "storage_mb": None, "embedding_mode": settings.embedding_mode,
        })
        backend_started = time.perf_counter()
        clickhouse.replace_vectors(dataset_id, dimension, _clickhouse_rows(rows))
        report.append({
            "dataset_id": dataset_id, "dimension": dimension, "backend": "clickhouse",
            "row_count": len(rows), "elapsed_s": round(time.perf_counter() - backend_started, 6),
            "storage_mb": round(clickhouse.storage_mb(dimension), 6), "embedding_mode": settings.embedding_mode,
        })
        backend_started = time.perf_counter()
        qdrant.replace_vectors(dataset_id, dimension, rows)
        report.append({
            "dataset_id": dataset_id, "dimension": dimension, "backend": "qdrant",
            "row_count": len(rows), "elapsed_s": round(time.perf_counter() - backend_started, 6),
            "storage_mb": None, "embedding_mode": settings.embedding_mode,
        })
    postgres.create_vector_indexes()
    _write_quantiles(dataset_id, bundle.quantile_frame)
    _write_report(report)
    return {
        "dataset_id": dataset_id,
        "segments": len(bundle.segments),
        "groundtruth": len(bundle.groundtruth),
        "dimensions": list(DIMENSIONS),
        "embedding_mode": settings.embedding_mode,
        "elapsed_s": round(time.perf_counter() - started, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a Faz 7 dataset into all enabled stores")
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    args = parser.parse_args()
    print(json.dumps(ingest(args.dataset), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
