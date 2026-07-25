"""Faz 2 sentetik olcek testi: gercek smoke embedding'lerin etrafina Gauss
gurultusu + gercekci filtre kolon dagilimiyla ayri bir bench_scale_<dim>
tablosuna N satir uretir. Uretim clips_* tablolarina DOKUNMAZ.

Kullanim: python scripts/build_scale_table.py --model xclip_hf_zeroshot --n 100000
Temizleme: DROP TABLE bench_scale_512  (bu script otomatik silmez)."""
import argparse
import json
import pathlib
import sys
import time

import clickhouse_connect
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

NOISE_STD = 0.05
BATCH_SIZE = 10_000
RNG_SEED = 1234

SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    video_id      LowCardinality(String),
    t_start       Float32,
    t_end         Float32,
    person_count  UInt16,
    car_count     UInt16,
    bus_count     UInt16,
    truck_count   UInt16,
    is_night      Bool,
    camera_motion Float32,
    brightness    Float32,
    platform      LowCardinality(String) DEFAULT 'bench_scale_synthetic',
    embedding     Array(Float32),
    INDEX idx_vec embedding TYPE vector_similarity('hnsw', 'cosineDistance', {dim})
        GRANULARITY 100000000,
    INDEX idx_bus bus_count TYPE minmax GRANULARITY 4,
    INDEX idx_person person_count TYPE minmax GRANULARITY 4
) ENGINE = MergeTree ORDER BY (video_id, t_start)
"""


def fetch_real_embeddings(ch, source_table: str):
    rows = ch.query(f"SELECT embedding FROM {source_table}").result_rows
    return np.array([r[0] for r in rows], dtype=np.float32)


def synth_batch(rng, real_embeddings, n, dim, start_idx):
    centers = real_embeddings[rng.integers(0, len(real_embeddings), size=n)]
    noise = rng.normal(0.0, NOISE_STD, size=(n, dim)).astype(np.float32)
    vecs = centers + noise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    # gercekci filtre dagilimi: satirlarin ~%5'i bus_count>=1 (plan varsayimi)
    has_bus = rng.random(n) < 0.05
    bus_count = np.where(has_bus, rng.integers(1, 3, size=n), 0)
    has_truck = rng.random(n) < 0.08
    truck_count = np.where(has_truck, rng.integers(1, 3, size=n), 0)
    person_count = rng.poisson(3.0, size=n).astype(np.int64)
    car_count = rng.poisson(8.0, size=n).astype(np.int64)
    is_night = rng.random(n) < 0.3
    camera_motion = rng.random(n).astype(np.float32)
    brightness = rng.uniform(50, 190, size=n).astype(np.float32)

    rows = []
    for i in range(n):
        idx = start_idx + i
        rows.append([
            f"synthetic_{idx:08d}", 0.0, 8.0,
            int(person_count[i]), int(car_count[i]), int(bus_count[i]), int(truck_count[i]),
            bool(is_night[i]), float(camera_motion[i]), float(brightness[i]),
            "bench_scale_synthetic", vecs[i].tolist(),
        ])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xclip_hf_zeroshot")
    ap.add_argument("--n", type=int, default=100_000)
    args = ap.parse_args()

    cfg = load_config()
    ch = clickhouse_connect.get_client(host=cfg["clickhouse"]["host"], port=cfg["clickhouse"]["port"])

    source_table = f"clips_{args.model}"
    real = fetch_real_embeddings(ch, source_table)
    if len(real) == 0:
        print(f"HATA: {source_table} bos, once ingest calistirin.")
        raise SystemExit(1)
    dim = real.shape[1]
    scale_table = f"bench_scale_{dim}"

    ch.command(SCHEMA_TEMPLATE.format(table=scale_table, dim=dim))
    ch.command(f"TRUNCATE TABLE {scale_table}")

    rng = np.random.default_rng(RNG_SEED)
    columns = ["video_id", "t_start", "t_end", "person_count", "car_count",
               "bus_count", "truck_count", "is_night", "camera_motion",
               "brightness", "platform", "embedding"]

    t_insert_start = time.perf_counter()
    inserted = 0
    while inserted < args.n:
        batch_n = min(BATCH_SIZE, args.n - inserted)
        rows = synth_batch(rng, real, batch_n, dim, inserted)
        ch.insert(scale_table, rows, column_names=columns)
        inserted += batch_n
        print(f"  {inserted}/{args.n} satir eklendi")
    insert_s = time.perf_counter() - t_insert_start

    t_merge_start = time.perf_counter()
    ch.command(f"OPTIMIZE TABLE {scale_table} FINAL")
    merge_s = time.perf_counter() - t_merge_start

    index_rows = ch.query(f"""
        SELECT name, formatReadableSize(data_compressed_bytes) AS size,
               data_compressed_bytes
        FROM system.data_skipping_indices
        WHERE table = '{scale_table}'
    """).result_rows

    count = ch.query(f"SELECT count() FROM {scale_table}").result_rows[0][0]

    summary = {
        "table": scale_table, "dim": dim, "n_rows": count,
        "insert_s": round(insert_s, 2), "optimize_final_s": round(merge_s, 2),
        "indices": [{"name": r[0], "size_readable": r[1], "size_bytes": r[2]} for r in index_rows],
        "cleanup_command": f"DROP TABLE {scale_table}",
    }
    out_path = pathlib.Path("artifacts/scale_table_build.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
