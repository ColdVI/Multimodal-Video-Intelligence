from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from app.config import Settings, settings
from app.ingestion.manifest import DatasetManifest, derive_identifier, load_manifest
from app.ingestion.telemetry import GenericCSVAdapter, align_timestamp


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    category: str
    status: str
    detail: str


@dataclass(frozen=True)
class SourcePair:
    video_id: str
    video_path: Path
    telemetry_path: Path | None
    video_start_unix_s: float | None = None
    offset_s: float | None = None


def _check(check_id: str, category: str, ok: bool, detail: str) -> PreflightCheck:
    return PreflightCheck(check_id, category, "pass" if ok else "fail", detail)


def _parse_start(value: str, timezone_name: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed.timestamp()


def discover_pairs(manifest: DatasetManifest, data_root: Path) -> tuple[SourcePair, ...]:
    root = data_root.expanduser().resolve()
    if manifest.pairing_strategy == "manifest_csv":
        mapping_path = (root / str(manifest.pairing_manifest_csv)).resolve()
        if not mapping_path.is_relative_to(root):
            raise ValueError("pairing manifest escapes DATA_ROOT")
        pairs: list[SourcePair] = []
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"video_id", "video_path", "telemetry_path"}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise ValueError(f"pairing manifest requires columns: {sorted(required)}")
            for row in reader:
                video_path = (root / row["video_path"]).resolve()
                telemetry_path = (root / row["telemetry_path"]).resolve() if row.get("telemetry_path") else None
                for candidate in (video_path, telemetry_path):
                    if candidate is not None and not candidate.is_relative_to(root):
                        raise ValueError(f"pairing path escapes DATA_ROOT: {candidate}")
                pairs.append(SourcePair(
                    video_id=str(row["video_id"]),
                    video_path=video_path,
                    telemetry_path=telemetry_path,
                    video_start_unix_s=(
                        None if not row.get("video_start_unix_s") else float(row["video_start_unix_s"])
                    ),
                    offset_s=None if not row.get("offset_s") else float(row["offset_s"]),
                ))
        return tuple(pairs)

    videos = manifest.resolve_glob(root, manifest.videos_glob)
    telemetry_paths = (
        manifest.resolve_glob(root, manifest.telemetry_glob)
        if manifest.telemetry_glob else ()
    )
    telemetry_by_id: dict[str, Path] = {}
    for path in telemetry_paths:
        telemetry_id = derive_identifier(path, manifest.telemetry_id_from)
        if telemetry_id in telemetry_by_id:
            raise ValueError(f"duplicate telemetry ID: {telemetry_id}")
        telemetry_by_id[telemetry_id] = path
    return tuple(
        SourcePair(
            video_id=derive_identifier(video_path, manifest.video_id_from),
            video_path=video_path,
            telemetry_path=telemetry_by_id.get(derive_identifier(video_path, manifest.video_id_from)),
        )
        for video_path in videos
    )


def _filename_start(manifest: DatasetManifest, path: Path) -> float | None:
    if manifest.video_start_time_from != "filename":
        return None
    if not manifest.filename_time_regex or not manifest.filename_time_format:
        raise ValueError("filename anchor requires filename_time_regex and filename_time_format")
    import re

    match = re.search(manifest.filename_time_regex, path.name)
    if match is None:
        raise ValueError(f"filename time regex did not match {path.name}")
    value = match.groupdict().get("timestamp") or match.group(0)
    parsed = datetime.strptime(value, manifest.filename_time_format)
    return parsed.replace(tzinfo=ZoneInfo(manifest.timezone)).timestamp()


def _segments_for_duration(duration_s: float, manifest: DatasetManifest) -> int:
    if duration_s <= 0:
        return 0
    if manifest.partial_window_policy == "pad_last":
        return max(1, math.ceil(duration_s / manifest.stride_s))
    if duration_s < manifest.window_size_s:
        return 0
    return math.floor((duration_s - manifest.window_size_s) / manifest.stride_s) + 1


def run_data_preflight(
    manifest_path: str | Path,
    *,
    data_root: Path,
    configured: Settings = settings,
    probe_fn: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    checks: list[PreflightCheck] = []
    try:
        manifest = load_manifest(manifest_path)
        checks.append(_check("manifest", "data", True, f"schema v{manifest.schema_version}; dataset={manifest.dataset_id}"))
    except Exception as exc:
        checks.append(_check("manifest", "data", False, f"{type(exc).__name__}: {exc}"))
        return _report(None, configured, checks, None, 0)

    checks.append(_check("relative_paths", "data", True, "all configured paths are DATA_ROOT-relative"))
    try:
        pairs = discover_pairs(manifest, data_root)
        checks.append(_check("video_count", "data", bool(pairs), f"videos={len(pairs)}"))
    except Exception as exc:
        checks.append(_check("pairing", "data", False, f"{type(exc).__name__}: {exc}"))
        return _report(manifest, configured, checks, None, 0)

    ids = [pair.video_id for pair in pairs]
    checks.append(_check("duplicate_video_ids", "data", len(ids) == len(set(ids)), f"unique={len(set(ids))}; total={len(ids)}"))
    if manifest.telemetry_glob or manifest.pairing_strategy == "manifest_csv":
        missing = [pair.video_id for pair in pairs if pair.telemetry_path is None or not pair.telemetry_path.is_file()]
        checks.append(_check("telemetry_pairing", "data", not missing, f"missing={missing[:10]}; total_missing={len(missing)}"))

    if probe_fn is None:
        try:
            from app.ingestion.video import probe_video

            probe_fn = probe_video
        except ImportError:
            probe_fn = None
    estimated_segments = 0
    sample_probe: Any = None
    if pairs and probe_fn is None:
        checks.append(PreflightCheck("video_probe", "data", "not_run", "video decoder is installed in Faz 11 Aşama 3"))
    elif pairs:
        try:
            sample_probe = probe_fn(pairs[0].video_path)  # type: ignore[misc]
            ok = sample_probe.duration_s > 0 and sample_probe.fps > 0 and sample_probe.width > 0 and sample_probe.height > 0
            checks.append(_check(
                "video_probe", "data", ok,
                f"duration_s={sample_probe.duration_s}; fps={sample_probe.fps}; size={sample_probe.width}x{sample_probe.height}",
            ))
            for pair in pairs:
                probe = sample_probe if pair is pairs[0] else probe_fn(pair.video_path)  # type: ignore[misc]
                estimated_segments += _segments_for_duration(float(probe.duration_s), manifest)
        except Exception as exc:
            checks.append(_check("video_probe", "data", False, f"{type(exc).__name__}: {exc}"))

    checks.append(_check(
        "clock_contract", "data", True,
        f"telemetry_clock={manifest.telemetry_clock}; offset_s={manifest.time_offset_s}; absolute={manifest.is_absolute_clock}",
    ))
    if pairs and pairs[0].telemetry_path and pairs[0].telemetry_path.is_file():
        try:
            start_anchor = pairs[0].video_start_unix_s
            if start_anchor is None:
                start_anchor = _filename_start(manifest, pairs[0].video_path)
            if start_anchor is None and sample_probe and sample_probe.creation_time:
                start_anchor = sample_probe.creation_time.timestamp()
            count = 0
            minimum = math.inf
            maximum = -math.inf
            previous: float | None = None
            monotonic = True
            for record in GenericCSVAdapter(manifest).iter_records(pairs[0].telemetry_path):
                value = align_timestamp(
                    record.timestamp,
                    telemetry_clock=manifest.telemetry_clock,
                    offset_s=pairs[0].offset_s if pairs[0].offset_s is not None else manifest.time_offset_s,
                    video_start_unix_s=start_anchor,
                )
                monotonic = monotonic and (previous is None or previous <= value)
                previous = value
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                count += 1
            checks.append(_check("timestamp_monotonicity", "data", monotonic and count > 0, f"records={count}"))
            if count and sample_probe:
                overlap = maximum >= 0 and minimum <= float(sample_probe.duration_s)
                checks.append(_check(
                    "telemetry_video_overlap", "data", overlap,
                    f"aligned=[{minimum:.3f},{maximum:.3f}]; video=[0,{sample_probe.duration_s:.3f}]",
                ))
        except Exception as exc:
            checks.append(_check("telemetry_alignment", "data", False, f"{type(exc).__name__}: {exc}"))

    checks.append(_check(
        "enabled_profile", "configuration", True,
        f"backends={configured.enabled_vector_backends}; dimensions={configured.enabled_dimensions}",
    ))
    vector_bytes = estimated_segments * sum(configured.enabled_dimensions) * 4 * len(configured.enabled_vector_backends)
    checks.append(_check(
        "storage_estimate", "resources", True,
        f"estimated_segments={estimated_segments}; vector_bytes={vector_bytes}; active_plus_staging_bytes={vector_bytes * 2}",
    ))
    checks.append(_check(
        "artifacts_access", "configuration", configured.artifacts_root.exists() and os.access(configured.artifacts_root, os.W_OK),
        f"path={configured.artifacts_root}; exists={configured.artifacts_root.exists()}",
    ))
    _append_model_checks(checks, configured)
    return _report(manifest, configured, checks, pairs, estimated_segments)


def _append_model_checks(checks: list[PreflightCheck], configured: Settings) -> None:
    requires_model = configured.embedding_mode in {"real", "hybrid_text"}
    bundle_root = configured.model_bundle_root.expanduser()
    if not bundle_root.is_dir():
        status = "fail" if requires_model else "not_run"
        checks.append(PreflightCheck(
            "model_bundle", "model", status,
            f"bundle missing at {bundle_root}; required_for_mode={requires_model}",
        ))
        checks.append(PreflightCheck(
            "qwen_import", "model", status,
            "Qwen import cannot be checked without a verified bundle",
        ))
        return
    try:
        from app.embedding.bundle import verify_bundle

        manifest = verify_bundle(
            bundle_root,
            expected_model_id=configured.qwen_model_id,
            expected_model_revision=configured.qwen_model_revision,
            expected_source_commit=configured.qwen_source_commit,
        )
        checks.append(_check(
            "model_bundle", "model", True,
            f"verified_bytes={manifest['total_size_bytes']}; source_commit={manifest['source_commit']}",
        ))
    except Exception as exc:
        checks.append(_check("model_bundle", "model", False, f"{type(exc).__name__}: {exc}"))
        checks.append(PreflightCheck("qwen_import", "model", "not_run", "bundle verification failed"))
        return
    repo_path = configured.qwen_repo_path
    if not repo_path.is_dir() and (bundle_root / "source").is_dir():
        repo_path = bundle_root / "source"
    try:
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        module = importlib.import_module("src.models.qwen3_vl_embedding")
        checks.append(_check(
            "qwen_import", "model", hasattr(module, "Qwen3VLEmbedder"), f"repo_path={repo_path}",
        ))
    except Exception as exc:
        checks.append(_check("qwen_import", "model", False, f"{type(exc).__name__}: {exc}"))


def _report(
    manifest: DatasetManifest | None,
    configured: Settings,
    checks: list[PreflightCheck],
    pairs: tuple[SourcePair, ...] | None,
    estimated_segments: int,
) -> dict[str, Any]:
    if any(item.status == "fail" for item in checks):
        status = "fail"
    elif any(item.status == "not_run" for item in checks):
        status = "not_run"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": None if manifest is None else manifest.dataset_id,
        "manifest_hash": None if manifest is None else manifest.manifest_hash,
        "video_count": 0 if pairs is None else len(pairs),
        "estimated_segments": estimated_segments,
        "enabled_backends": list(configured.enabled_vector_backends),
        "enabled_dimensions": list(configured.enabled_dimensions),
        "checks": [asdict(item) for item in checks],
    }


def exit_code(report: dict[str, Any]) -> int:
    failed = {item["category"] for item in report["checks"] if item["status"] == "fail"}
    for category, code in (
        ("configuration", 2), ("data", 3), ("gpu", 4), ("model", 5), ("resources", 6),
    ):
        if category in failed:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only container dataset preflight")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = run_data_preflight(args.dataset, data_root=args.data_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
