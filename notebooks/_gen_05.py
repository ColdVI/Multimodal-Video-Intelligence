"""notebooks/05_hybrid_query_benchmark.ipynb'i insa eder. Spec SS4.6 + SS7
(Blok C - backend dogruluk/gecikme). notebook 04'un GERCEKTEN 'healthy'
bulundugu backend'ler + notebook 02'nin GERCEK embedding'leri olmadan
BENCHMARK KOSULMAZ - bos backend/embedding durumunda bu notebook ACIKCA
'kanit yok' yazar, sayi UYDURMAZ."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute
from src.research.colab_paths import COLAB_BOOTSTRAP_CELL

CELLS = [
("code", COLAB_BOOTSTRAP_CELL),
("md", """# 05 - Hybrid sorgu benchmarki (Blok C-F, CPU/high-RAM asamasi)

Spec SS4.6 + SS7. **Onkosul:** notebook 04'un `ingest_report.csv`'si ve
manifest'indeki `backend_statuses` en az bir backend icin `'healthy'`
olmali, notebook 02'nin embedding checkpoint'leri Drive'da olmali. Ikisi de
yoksa bu notebook GERCEKTEN BENCHMARK KOSMAZ - durumu acikca yazar."""),

("code", """import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research import colab_paths
from src.research.common_exact import exact_ranking, recall_at_k_vs_exact, topk_agreement
from src.research.config import DEFAULT as cfg
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest

OUT = colab_paths.research_root()
hw = detect_hardware_profile()
print(f"OUT={OUT}  drive_mounted={colab_paths.drive_mounted()}")
"""),

("md", "## On-kosul kontrolu (SS11: sessiz basarisizlik yok)"),

("code", """nb04_manifest_path = OUT / "04_vector_backend_loading_manifest.json"
backend_statuses = {}
if nb04_manifest_path.exists():
    nb04_manifest = json.loads(nb04_manifest_path.read_text(encoding="utf-8"))
    backend_statuses = nb04_manifest["extra"]["backend_statuses"]
healthy_backends = [b for b, s in backend_statuses.items() if s == "healthy"]
print(f"notebook 04 backend durumlari: {backend_statuses}")
print(f"saglikli backend'ler: {healthy_backends}")

auair_seg_path = OUT / "auair_segments.parquet"
selectivity_path = OUT / "selectivity_thresholds.json"
EMB_ROOT = colab_paths.embeddings_root()
auair_emb_2048_path = EMB_ROOT / "auair_qwen2048.json"

can_run_real_benchmark = bool(healthy_backends) and auair_emb_2048_path.exists() and auair_seg_path.exists()
print(f"\\ncan_run_real_benchmark={can_run_real_benchmark}")
if not can_run_real_benchmark:
    reasons = []
    if not healthy_backends:
        reasons.append("hicbir backend 'healthy' degil (notebook 04)")
    if not auair_emb_2048_path.exists():
        reasons.append(f"{auair_emb_2048_path} yok (notebook 02 GPU asamasi calismamis)")
    if not auair_seg_path.exists():
        reasons.append(f"{auair_seg_path} yok (notebook 01 calismamis)")
    print("BENCHMARK KOSULAMIYOR, nedenler: " + "; ".join(reasons))
"""),

("md", "## SS6 - common exact referans (numpy, backend'siz) - HER ZAMAN hesaplanabilir"),

("code", """common_exact_ready = False
if auair_emb_2048_path.exists() and auair_seg_path.exists():
    embeddings_by_id = json.loads(auair_emb_2048_path.read_text(encoding="utf-8"))
    seg_df = pd.read_parquet(auair_seg_path)
    ids_with_emb = [sid for sid in seg_df["segment_id"] if sid in embeddings_by_id]
    print(f"{len(ids_with_emb)}/{len(seg_df)} segment icin 2048d embedding mevcut.")
    if ids_with_emb:
        corpus = np.array([embeddings_by_id[i] for i in ids_with_emb], dtype=np.float32)
        print(f"common_exact referans matrisi hazir: shape={corpus.shape}")
        common_exact_ready = True
else:
    print("[ATLANDI] AU-AIR embedding/segment verisi yok - common_exact referansi kurulamadi.")
"""),

("md", "## Blok C - backend dogruluk/gecikme (SS7.3) - yalniz saglikli backend + gercek embedding varsa"),

("code", """import time

from src.research.backends import ch, pv, qd

SELECTIVITY_LEVELS = (1.0, 0.5, 0.1, 0.01, 0.001)
WARMUP = cfg.warmup
REPS = cfg.measured_repetitions
FINAL_K = cfg.top_k
N_QUERIES = 20  # spec SS8: 20 sabit sorgu, tum backend/boyutlarda ayni

bench_rows = []
if not can_run_real_benchmark:
    print("[BENCHMARK KOSULMADI] Yukaridaki on-kosul kontrolu basarisiz - "
         "vector_database_results.csv BOS/durum-etiketli yazilacak (sayi UYDURULMAYACAK).")
else:
    selectivity_thresholds = json.loads(selectivity_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(cfg.random_seed)
    query_ids = list(rng.choice(ids_with_emb, size=min(N_QUERIES, len(ids_with_emb)), replace=False))
    exact_order_full = {qid: exact_ranking(corpus, np.array(embeddings_by_id[qid], dtype=np.float32))
                        for qid in query_ids}

    def _run_one_query(backend, dim, table, qvec, where, qd_filter):
        t0 = time.perf_counter()
        if backend == "clickhouse":
            client = ch.get_client()
            sql = ch.build_query_sql(table, qvec, FINAL_K, where=where)
            rows = client.query(sql).result_rows
            ids = [r[0] for r in rows]
        elif backend == "qdrant":
            client = qd.get_client()
            hits = qd.query(client, table, qvec, FINAL_K, filter_conditions=qd_filter)
            ids = [h.payload["segment_id"] for h in hits]
        elif backend == "pgvector":
            conn = pv.get_connection()
            sql = pv.build_query_sql(table, qvec, FINAL_K, where=where)
            with conn.cursor() as cur:
                cur.execute(sql)
                ids = [r[0] for r in cur.fetchall()]
        else:
            raise ValueError(f"bilinmeyen backend: {backend}")
        latency = time.perf_counter() - t0
        return ids, latency

    for backend in healthy_backends:
        for dim in cfg.dims:
            table = f"seg_{dim}"
            for selectivity in SELECTIVITY_LEVELS:
                # NOT: tablo/koleksiyon artik TUM dataset'leri tutuyor (notebook 04
                # duzeltmesi) - exact-referans (corpus) SADECE AU-AIR'den kuruldugu
                # icin sorgu da dataset_id='auair' ile filtrelenmeli, yoksa ANN
                # sonucu diger dataset'lerden komsu dondurup recall/agreement'i
                # anlamsizlastirir.
                qd_filter = {"dataset_id": "auair"}
                if selectivity >= 1.0:
                    where = "dataset_id = 'auair'"
                else:
                    theta = selectivity_thresholds['altitude_m'][str(selectivity)]['threshold']
                    where = f"dataset_id = 'auair' AND altitude_m < {theta}"
                    qd_filter["altitude_m"] = {"lt": theta}
                for _ in range(WARMUP):
                    qid = query_ids[0]
                    _run_one_query(backend, dim, table, embeddings_by_id[qid], where, qd_filter)
                latencies, returned_counts, recalls, agreements = [], [], [], []
                for qid in query_ids[:REPS] if REPS <= len(query_ids) else query_ids:
                    qvec = embeddings_by_id[qid]
                    ids, latency = _run_one_query(backend, dim, table, qvec, where, qd_filter)
                    latencies.append(latency * 1000)
                    returned_counts.append(len(ids))
                    id_to_idx = {sid: i for i, sid in enumerate(ids_with_emb)}
                    backend_idx = [id_to_idx[i] for i in ids if i in id_to_idx]
                    recalls.append(recall_at_k_vs_exact(backend_idx, exact_order_full[qid], FINAL_K))
                    exact_top_ids = [ids_with_emb[i] for i in exact_order_full[qid][:FINAL_K]]
                    agreements.append(topk_agreement(ids, exact_top_ids, FINAL_K))
                lat_sorted = sorted(latencies)
                bench_rows.append({
                    "backend": backend, "dimension": dim, "selectivity_target": selectivity,
                    "returned_count": int(np.mean(returned_counts)),
                    "underfilled": bool(np.mean(returned_counts) < FINAL_K),
                    "topk_agreement": float(np.mean(agreements)),
                    "ann_recall_vs_exact": float(np.mean(recalls)),
                    "p50_ms": float(np.median(lat_sorted)),
                    "p95_ms": float(np.percentile(lat_sorted, 95)),
                    "n_queries": len(latencies),
                    "hardware_profile": hw["hardware_profile"],
                })
                print(f"  {backend} d={dim} sel={selectivity}: "
                     f"recall={bench_rows[-1]['ann_recall_vs_exact']:.2f} p50={bench_rows[-1]['p50_ms']:.1f}ms")
    print(f"\\n{len(bench_rows)} satir uretildi (healthy_backends={healthy_backends}).")
"""),

("md", "## Ozet (vector_database_results.csv, bench_raw.parquet - SS4.6/SS10)"),

("code", """bench_df = pd.DataFrame(bench_rows) if bench_rows else pd.DataFrame(
    columns=["backend", "dimension", "selectivity_target", "returned_count", "underfilled",
            "topk_agreement", "ann_recall_vs_exact", "p50_ms", "p95_ms", "hardware_profile"])
results_path = OUT / "vector_database_results.csv"
bench_df.to_csv(results_path, index=False)
raw_path = OUT / "bench_raw.parquet"
bench_df.to_parquet(raw_path, index=False)

status = "REAL" if bench_rows else "KANIT_YOK"
report_md = f'''# Notebook 05 - Hybrid sorgu benchmarki sonucu

## Durum: {status}

- can_run_real_benchmark={can_run_real_benchmark}
- healthy_backends (notebook 04'ten)={healthy_backends}
- common_exact_ready={common_exact_ready}
- uretilen satir sayisi={len(bench_df)}

{"Bu ortamda (Colab disi/backend kurulamadi) benchmark KOSULMADI. "
 "vector_database_results.csv BOS yazildi - sifir/varsayilan deger UYDURULMADI, "
 "dosya kendisi 'veri yok' anlamina gelir." if not bench_rows else
 "Benchmark GERCEKTEN kosuldu, sonuclar asagida."}
'''
(OUT / "vector_database_report.md").write_text(report_md, encoding="utf-8")
print(report_md)
print(f"-> {results_path} ({len(bench_df)} satir)")

manifest = RunManifest(
    notebook="05_hybrid_query_benchmark",
    hardware_profile=hw["hardware_profile"],
    extra={
        "can_run_real_benchmark": can_run_real_benchmark,
        "healthy_backends": healthy_backends,
        "common_exact_ready": common_exact_ready,
        "n_bench_rows": len(bench_df),
        "status": status,
    },
)
manifest_path = write_manifest(manifest, OUT)
print(f"\\nmanifest -> {manifest_path}")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/05_hybrid_query_benchmark.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=300)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
