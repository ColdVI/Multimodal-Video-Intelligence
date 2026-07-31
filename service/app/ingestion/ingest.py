from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from app.config import settings
from app.db import postgres
from app.db.ingest_runs import ChunkSpec, PostgresRunStore, RunCoordinator, RunSpec, new_run_spec
from app.db.registry import BACKEND_REGISTRY, initialize_enabled_backends
from app.ingestion.generic_loader import WindowRecord, batched, iter_window_records, release_frames
from app.ingestion.manifest import DatasetManifest, load_manifest
from app.mrl import truncate_and_normalize
from app.preflight import run_data_preflight


def _chunk_hash(records: Iterable[WindowRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.segment_id.encode("utf-8"))
        digest.update(f"{record.t_start:.6f}:{record.t_end:.6f}".encode("ascii"))
    return digest.hexdigest()


def _vector_rows(records: list[WindowRecord], vectors: np.ndarray, dimension: int) -> list[dict[str, Any]]:
    rows = []
    for record, base in zip(records, vectors, strict=True):
        vector = truncate_and_normalize(base, dimension)
        telemetry = record.telemetry
        rows.append({
            "segment_id": record.segment_id, "dataset_id": record.dataset_id,
            "video_id": record.video_id, "t_start": record.t_start, "t_end": record.t_end,
            "event_category": telemetry.get("event_category"), "split": telemetry.get("split"),
            "latitude": telemetry.get("latitude"), "longitude": telemetry.get("longitude"),
            "altitude_m": telemetry.get("altitude_m"), "velocity_mps": telemetry.get("velocity_mps"),
            "roll": telemetry.get("roll"), "pitch": telemetry.get("pitch"), "yaw": telemetry.get("yaw"),
            "yaw_rate": telemetry.get("yaw_rate"), "gimbal_pitch": telemetry.get("gimbal_pitch"),
            "gimbal_heading": telemetry.get("gimbal_heading"),
            "compass_heading": telemetry.get("compass_heading"),
            "person_count": int(telemetry.get("person_count") or 0),
            "vehicle_count": int(telemetry.get("vehicle_count") or 0),
            "bus_count": int(telemetry.get("bus_count") or 0),
            "is_night": int(bool(telemetry.get("is_night"))),
            "embedding": vector.astype(np.float32, copy=False).tolist(),
        })
    return rows


def _metadata_rows(records: list[WindowRecord]) -> tuple[list[tuple], list[tuple], list[tuple], list[tuple]]:
    videos: dict[str, tuple] = {}
    segments, metadata, telemetry = [], [], []
    for record in records:
        values = record.telemetry
        videos[record.video_id] = (
            record.dataset_id, record.video_id, record.metadata["source_path"], values.get("split"),
            record.metadata.get("video_duration_s"), values.get("event_category"),
        )
        segments.append((
            record.segment_id, record.dataset_id, record.video_id, record.t_start, record.t_end, None,
        ))
        metadata.append((
            record.segment_id, int(values.get("person_count") or 0), int(values.get("vehicle_count") or 0),
            int(values.get("bus_count") or 0), [], None, None,
        ))
        telemetry.append((
            record.segment_id, None, None, values.get("latitude"), values.get("longitude"),
            values.get("altitude_m"), values.get("velocity_mps"), values.get("roll"),
            values.get("pitch"), values.get("yaw"), values.get("yaw_rate"),
            values.get("gimbal_pitch"), values.get("gimbal_heading"), values.get("compass_heading"),
            None, json.dumps({**record.extra, "is_night": bool(values.get("is_night"))}),
        ))
    return list(videos.values()), segments, metadata, telemetry


class GenericIngestor:
    def __init__(
        self,
        manifest: DatasetManifest,
        data_root: Path,
        run: RunSpec,
        *,
        store: Any,
        backends: dict[str, Any],
        embed_videos: Callable[[list[Any]], np.ndarray],
        records: Iterable[WindowRecord] | None = None,
        report_root: Path | None = None,
    ):
        self.manifest = manifest
        self.data_root = data_root
        self.run_spec = run
        self.store = store
        self.backends = backends
        self.embed_videos = embed_videos
        self.records = records
        self.report_root = report_root or settings.artifacts_root / "faz11" / "ingest_runs" / run.run_id
        self.coordinator = RunCoordinator(store, backends)

    def _flush_pending(
        self,
        chunk_index: int,
        pending: list[tuple[WindowRecord, np.ndarray]],
        backend_rows: dict[str, int],
    ) -> int:
        """Write buffered (record, vector) pairs to Postgres metadata and every enabled
        vector backend. Called once per DB_WRITE_BATCH_SIZE slice, independent of the
        embed batch size that produced the vectors."""
        if not pending:
            return 0
        records = [record for record, _ in pending]
        vectors = np.stack([vector for _, vector in pending])
        video_rows, segment_rows, metadata_rows, telemetry_rows = _metadata_rows(records)
        count = postgres.write_run_metadata_chunk(
            self.run_spec.run_id, self.manifest.dataset_id, chunk_index,
            video_rows, segment_rows, metadata_rows, telemetry_rows,
        )
        for dimension in self.run_spec.enabled_dimensions:
            rows = _vector_rows(records, vectors, dimension)
            for backend_name in self.run_spec.enabled_backends:
                backend_rows[f"{backend_name}:{dimension}"] += self.backends[backend_name].write_chunk(
                    self.run_spec.run_id, self.manifest.dataset_id, dimension, chunk_index, rows,
                )
        return count

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        self.report_root.mkdir(parents=True, exist_ok=True)
        errors_path = self.report_root / "errors.jsonl"
        committed = set()
        if resume:
            committed = {
                (str(row["video_id"]), int(row["chunk_index"]))
                for row in self.store.chunks(self.run_spec.run_id) if row["status"] == "committed"
            }
        if self.records is not None:
            iterator = (
                record for record in self.records
                if (record.video_id, record.chunk_index) not in committed
            )
        else:
            iterator = iter_window_records(
                self.manifest, data_root=self.data_root, committed_chunks=committed,
            )
        self.store.set_run_status(self.run_spec.run_id, "ingesting")
        chunk_reports = []
        total_segments = sum(int(row["expected_segments"]) for row in self.store.chunks(self.run_spec.run_id) if row["status"] == "committed")
        failed = False
        for (video_id, chunk_index), group in itertools.groupby(iterator, key=lambda row: (row.video_id, row.chunk_index)):
            group_iterator = iter(group)
            first = next(group_iterator)
            source_path = str(first.metadata["source_path"])
            provisional = ChunkSpec(
                run_id=self.run_spec.run_id, dataset_id=self.manifest.dataset_id, video_id=video_id,
                video_path=source_path, chunk_index=chunk_index,
                chunk_start_s=chunk_index * settings.decode_chunk_s,
                chunk_end_s=min(float(first.metadata["video_duration_s"]), (chunk_index + 1) * settings.decode_chunk_s),
                expected_segments=0,
            )
            try:
                self.coordinator.begin_chunk(provisional, self.run_spec)
                postgres.delete_inactive_metadata_chunk(
                    self.run_spec.run_id, self.manifest.dataset_id, video_id, chunk_index,
                )
                backend_rows = {
                    f"{backend}:{dimension}": 0
                    for backend in self.run_spec.enabled_backends for dimension in self.run_spec.enabled_dimensions
                }
                metadata_count = 0
                chunk_records_for_hash: list[WindowRecord] = []
                pending: list[tuple[WindowRecord, np.ndarray]] = []
                # Three independently-sized stages, per FAZ11 spec §6.2/§6.4:
                # decode_prefetch_windows bounds how many decoded (frame-holding) windows
                # are pulled off the lazy decode iterator at once (RAM bound); embed_batch_size
                # is the Qwen call granularity (VRAM bound); db_write_batch_size is the
                # Postgres/vector-backend write granularity (DB/network bound). None of the
                # three may be collapsed into another.
                decode_source = itertools.chain([first], group_iterator)
                for prefetch_group in batched(decode_source, settings.decode_prefetch_windows):
                    for embed_batch in batched(prefetch_group, settings.embed_batch_size):
                        try:
                            vectors = self.embed_videos([record.frames for record in embed_batch])
                            if vectors.shape != (len(embed_batch), 2048) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                                raise RuntimeError(f"invalid Qwen batch output: shape={vectors.shape}; dtype={vectors.dtype}")
                            pending.extend(zip(embed_batch, vectors))
                            chunk_records_for_hash.extend(embed_batch)
                        finally:
                            release_frames(embed_batch)
                        while len(pending) >= settings.db_write_batch_size:
                            write_batch, pending = pending[:settings.db_write_batch_size], pending[settings.db_write_batch_size:]
                            metadata_count += self._flush_pending(chunk_index, write_batch, backend_rows)
                metadata_count += self._flush_pending(chunk_index, pending, backend_rows)
                pending = []
                completed_spec = replace(provisional, expected_segments=metadata_count)
                statuses = self.coordinator.commit_chunk(
                    completed_spec, self.run_spec, backend_rows, metadata_rows=metadata_count,
                )
                total_segments += metadata_count
                chunk_reports.append({
                    "video_id": video_id, "chunk_index": chunk_index, "segments": metadata_count,
                    "chunk_hash": _chunk_hash(chunk_records_for_hash), "backend_status": statuses,
                })
            except Exception as exc:
                failed = True
                self.store.set_chunk_status(provisional, "failed", {"error": str(exc)})
                with errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "run_id": self.run_spec.run_id, "dataset_id": self.manifest.dataset_id,
                        "video_id": video_id, "chunk_index": chunk_index,
                        "error_type": type(exc).__name__, "error": str(exc),
                    }, ensure_ascii=False) + "\n")
                if self.manifest.fail_on_video_error:
                    break
        final_spec = replace(self.run_spec, expected_segments=total_segments)
        self.store.update_expected_segments(final_spec.run_id, total_segments)
        result = self.coordinator.finalize_run(final_spec) if not failed else {
            "status": "failed", "run_id": final_spec.run_id, "active_changed": False,
        }
        if failed:
            self.store.set_run_status(final_spec.run_id, "failed", error_summary={"errors_path": str(errors_path)})
        report = {
            "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": result["status"], "run": asdict(final_spec), "resume": resume,
            "preexisting_committed_chunks": len(committed), "segments": total_segments,
            "chunks": chunk_reports, "elapsed_s": round(time.perf_counter() - started, 6),
            "errors_path": str(errors_path), "finalize": result,
        }
        (self.report_root / "report.json").write_text(
            json.dumps(report, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report


def ingest_manifest(path: Path, *, data_root: Path, resume: bool) -> dict[str, Any]:
    if settings.embedding_mode != "real":
        raise ValueError("generic manifest ingest requires EMBEDDING_MODE=real; no synthetic fallback is allowed")
    manifest = load_manifest(path)
    preflight = run_data_preflight(path, data_root=data_root)
    if preflight["status"] != "pass":
        raise RuntimeError(f"data/model preflight did not pass: {preflight['status']}")
    postgres.init_schema(include_vectors=False)
    initialize_enabled_backends()
    store = PostgresRunStore()
    run = store.find_resumable(manifest.dataset_id, manifest.manifest_hash) if resume else None
    if run is None:
        run = new_run_spec(
            dataset_id=manifest.dataset_id, dataset_version=manifest.manifest_hash[:12],
            vector_provenance="real", model_id=settings.qwen_model_id,
            model_revision=settings.qwen_model_revision, source_commit=settings.qwen_source_commit,
            enabled_backends=settings.enabled_vector_backends, enabled_dimensions=settings.enabled_dimensions,
            manifest_hash=manifest.manifest_hash, expected_segments=None,
        )
        store.create(run, status="preflight_passed")
    from app.db.telemetry_registry import replace_fields
    from app.embedding.qwen import embed_videos
    from app.search.filter_schema import manifest_filter_fields

    replace_fields(manifest.dataset_id, run.run_id, manifest_filter_fields(manifest))

    backends = {name: BACKEND_REGISTRY[name] for name in run.enabled_backends}
    return GenericIngestor(
        manifest, data_root, run, store=store, backends=backends, embed_videos=embed_videos,
    ).run(resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run legacy or generic manifest ingestion")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="generic dataset manifest")
    source.add_argument("--dataset-id", choices=("auair", "capera", "seadronessee"), help="legacy loader")
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dataset_id:
        from app.ingestion.load_dataset import ingest as legacy_ingest

        result = legacy_ingest(args.dataset_id)
    else:
        result = ingest_manifest(args.dataset, data_root=args.data_root, resume=args.resume)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
