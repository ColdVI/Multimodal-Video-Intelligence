from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.db import clickhouse, postgres, qdrant
from app.embedding.router import embed_item
from app.mrl import SUPPORTED_DIMS


def _clean(value: Any) -> Any:
    if value is None: return None
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError):
        pass
    return value


def _source_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_auair(root: Path) -> tuple[dict, list[dict]]:
    seg_path = root / "research/auair_segments.parquet"
    tel_path = root / "research/auair_telemetry.parquet"
    segments, telemetry = pd.read_parquet(seg_path), pd.read_parquet(tel_path)
    tel = {str(row["segment_id"]): row for row in telemetry.to_dict("records")}
    rows = []
    for seg in segments.to_dict("records"):
        item = tel.get(str(seg["segment_id"]), {})
        altitude = _clean(item.get("altitude_m"))
        # Parquet contract says metres. A median over 500 is a defensive signal
        # that an older notebook left millimetres in place.
        if altitude is not None and altitude > 500: altitude = float(altitude) / 1000.0
        rows.append({
            "segment_id": str(seg["segment_id"]), "dataset_id":"auair", "video_id":str(seg["video_id"]),
            "t_start":float(seg["t_start"]), "t_end":float(seg["t_end"]), "caption":None,
            "file_path":None, "split":None, "event_category":None, "has_telemetry":True,
            "altitude_m":altitude, "velocity_mps":_clean(item.get("velocity_mps")),
            "roll":_clean(item.get("roll")), "pitch":_clean(item.get("pitch")), "yaw":_clean(item.get("yaw")),
            "yaw_rate":_clean(item.get("yaw_rate")), "gimbal_pitch":None, "person_count":int(item.get("person_count",0)),
            "vehicle_count":int(item.get("vehicle_count",0)), "bus_count":0,
        })
    dataset = {"dataset_id":"auair","dataset_version":"verified-1866","source_hash":_source_hash([seg_path,tel_path]),"license":"CC BY-NC-SA","has_telemetry":True,"has_captions":False}
    return dataset, rows


def load_capera(root: Path) -> tuple[dict, list[dict]]:
    candidates = list((settings.data_dir / "downloads").glob("**/*caption*.json"))
    if not candidates:
        raise FileNotFoundError("CapERA caption JSON not found under data/downloads")
    path = candidates[0]
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("videos", raw.get("annotations", []))
    rows = []
    for record in records:
        video_id = str(record.get("video_id", record.get("id", record.get("video", ""))))
        captions = record.get("captions", [record.get("caption")])
        caption = next((str(x) for x in captions if x), None)
        rows.append({"segment_id":f"capera:{video_id}:0.000:5.000","dataset_id":"capera","video_id":video_id,"t_start":0.0,"t_end":5.0,"caption":caption,"file_path":record.get("path"),"split":record.get("split"),"event_category":record.get("event_category"),"has_telemetry":False})
    return {"dataset_id":"capera","dataset_version":"local","source_hash":_source_hash([path]),"license":"dataset terms","has_telemetry":False,"has_captions":True}, rows


def load_seadronessee(root: Path) -> tuple[dict, list[dict]]:
    candidates = list((settings.data_dir / "downloads").glob("**/*SeaDronesSee*.json"))
    if not candidates: raise FileNotFoundError("SeaDronesSee annotation JSON not found under data/downloads")
    path = candidates[0]; raw = json.loads(path.read_text(encoding="utf-8")); images = raw.get("images", [])
    rows = []
    for image in images:
        try:
            source, meta = image["source"], image["meta"]
            video_id, frame = str(image["video_id"]), int(image["frame_index"])
            t = frame / 30.0
            rows.append({"segment_id":f"seadronessee:{video_id}:{t:.3f}:{t+1/30:.3f}","dataset_id":"seadronessee","video_id":video_id,"t_start":t,"t_end":t+1/30,"caption":None,"file_path":source.get("video"),"has_telemetry":True,"altitude_m":meta.get("altitude"),"velocity_mps":meta.get("speed"),"gimbal_pitch":meta.get("gimbal_pitch"),"gimbal_heading":meta.get("gimbal_heading"),"compass_heading":meta.get("compass_heading"),"person_count":0,"vehicle_count":0,"bus_count":0})
        except (KeyError,TypeError,ValueError) as exc:
            _error({"dataset":"seadronessee","record":image,"error":str(exc)})
    return {"dataset_id":"seadronessee","dataset_version":"local","source_hash":_source_hash([path]),"license":"official dataset terms","has_telemetry":True,"has_captions":False}, rows


def _error(record: dict) -> None:
    path = settings.artifacts_dir / "research/ingest_errors.jsonl"; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")


LOADERS = {"auair":load_auair,"capera":load_capera,"seadronessee":load_seadronessee}


def derive_selectivity(rows: list[dict]) -> dict:
    result = {}
    directions = {"altitude_m":"less_than","velocity_mps":"greater_than","gimbal_pitch":"greater_than","person_count":"greater_than"}
    for field, direction in directions.items():
        values = np.asarray([r[field] for r in rows if r.get(field) is not None],dtype=np.float64)
        levels = {}
        for p in (0.5,0.1,0.01,0.001):
            if not len(values): levels[str(p)]={"threshold":None,"actual_selectivity":None,"n":0}; continue
            q = p if direction == "less_than" else 1-p
            threshold=float(np.quantile(values,q)); actual=float(np.mean(values < threshold if direction=="less_than" else values > threshold))
            levels[str(p)]={"threshold":threshold,"actual_selectivity":actual,"n":int(len(values))}
        result[field]={"direction":direction,"levels":levels}
    return result


def ingest(dataset_id: str) -> dict:
    if dataset_id not in LOADERS: raise ValueError(f"unknown dataset: {dataset_id}")
    dataset, rows = LOADERS[dataset_id](settings.artifacts_dir)
    if not rows: raise ValueError(f"no rows parsed for {dataset_id}")
    postgres.init_schema(); clickhouse.init_schema(); qdrant.init_schema()
    started=time.perf_counter(); postgres.upsert_dataset(dataset); postgres.upsert_segments(rows)
    vectors: dict[int,list] = {d:[] for d in SUPPORTED_DIMS}
    for row in rows:
        for dim in SUPPORTED_DIMS:
            vector=embed_item(dataset_id,row["segment_id"],row.get("file_path"),dim)
            if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector))-1)>1e-5: raise AssertionError(row["segment_id"])
            vectors[dim].append((row["segment_id"],vector))
    timings=[]
    for backend, fn in (("pgvector",lambda:postgres.upsert_embeddings(dataset_id,vectors)),("clickhouse",lambda:clickhouse.insert_segments(rows,vectors)),("qdrant",lambda:qdrant.upsert(rows,vectors))):
        mark=time.perf_counter(); fn(); elapsed=time.perf_counter()-mark
        for dim in SUPPORTED_DIMS: timings.append({"dataset_id":dataset_id,"dimension":dim,"backend":backend,"row_count":len(rows),"duration_s":round(elapsed,3),"storage_mb":None})
    postgres.ensure_hnsw_indexes()
    thresholds=derive_selectivity(rows); (settings.artifacts_dir/"research/selectivity_thresholds_v2.json").write_text(json.dumps(thresholds,indent=2),encoding="utf-8")
    report=settings.artifacts_dir/"research/ingest_report.csv"; report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=timings[0]); writer.writeheader(); writer.writerows(timings)
    return {"dataset_id":dataset_id,"rows":len(rows),"embedding_mode":settings.embedding_mode,"duration_s":round(time.perf_counter()-started,3)}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",required=True,choices=sorted(LOADERS)); args=parser.parse_args()
    print(json.dumps(ingest(args.dataset),ensure_ascii=False))


if __name__ == "__main__": main()
