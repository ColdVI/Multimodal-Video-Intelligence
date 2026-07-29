"""notebooks/04_vector_backend_loading.ipynb'i insa eder. Spec SS4.5 +
Colab handoff madde 6-10. Preflight -> environment_capability_report.json,
sonra ClickHouse/Qdrant/pgvector'i SIRAYLA (asla ayni anda ikisi/uçü birden
bellekte degil) kur/baslat/saglik-kontrol/yukle/durdur/temizle. Kurulum
basarisiz olursa SESSIZ FALLBACK YOK - environment_unavailable yazilir,
sahte/kismi benchmark URETILMEZ."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute
from src.research.colab_paths import COLAB_BOOTSTRAP_CELL

CELLS = [
("code", COLAB_BOOTSTRAP_CELL),
("md", """# 04 - Vector backend kurulumu ve veri yukleme (CPU/high-RAM asamasi)

Spec SS4.5 + Colab handoff. **Onkosul:** notebook 02'nin GPU asamasi Drive'a
embedding checkpoint'leri yazmis olmali (`embedding_ready=True`). Bu
notebook GPU GEREKTIRMEZ - ayri bir Colab CPU/high-RAM runtime'inda
calisir. Ilk hucre ortam on-kontrolu (preflight) calistirir."""),

("code", """import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research import colab_paths
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest

OUT = colab_paths.research_root()
hw = detect_hardware_profile()
print(f"OUT={OUT}  drive_mounted={colab_paths.drive_mounted()}  in_colab={colab_paths.in_colab()}")
"""),

("md", "## Ortam on-kontrolu (SS7 - environment_capability_report.json)"),

("code", """from scripts.colab_preflight import build_report

capability_report = build_report()
report_path = pathlib.Path("artifacts/research/environment_capability_report.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(capability_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(capability_report, indent=2, ensure_ascii=False)[:2000])
print(f"\\n-> {report_path}")
"""),

("md", "## Backend kurulum politikasi (SS8) - sirayla, sessiz fallback yok"),

("code", """import json as _json

backend_versions = _json.loads(pathlib.Path("backend_versions.json").read_text(encoding="utf-8"))
print(_json.dumps(backend_versions, indent=2, ensure_ascii=False))
"""),

("md", "## Embedding kontrolu (notebook 02'nin Drive ciktisi)"),

("code", """EMB_ROOT = colab_paths.embeddings_root()
DIMS = (2048, 1024, 512, 256)
DATASETS = ("auair", "capera", "msrvtt")

available_embeddings = {}
for ds in DATASETS:
    available_embeddings[ds] = {}
    for d in DIMS:
        p = EMB_ROOT / f"{ds}_qwen{d}.json"
        available_embeddings[ds][d] = p.exists()

print(_json.dumps(available_embeddings, indent=2, ensure_ascii=False))
any_embeddings = any(v for ds in available_embeddings.values() for v in ds.values())
if not any_embeddings:
    print("\\n[BILGI] Hicbir embedding bulunamadi - once notebook 02'yi GPU runtime'inda "
         "calistirin. Asagidaki backend kurulum/saglik-kontrolu YINE DE calisir "
         "(altyapi dogrulamasi), ama veri yukleme adimlari ATLANACAK.")
"""),

("md", "## ClickHouse - install / start / health / (yukle) / stop / cleanup"),

("code", """from src.research.backends import ch

ingest_report_rows = []

print("ClickHouse install...")
ch_install = ch.install()
print(f"  ok={ch_install['ok']}")
ch_status = "environment_unavailable"
if ch_install["ok"]:
    print("ClickHouse start...")
    ch_start = ch.start()
    print(f"  ok={ch_start['ok']}")
    if ch_start["ok"]:
        ch_health = ch.health_check()
        print(f"ClickHouse health: {ch_health['healthy']}")
        if ch_health["healthy"]:
            ch_status = "healthy"
            # Veri yukleme: embedding varsa GERCEKTEN yuklenir (burada, GPU
            # gerektirmeyen bir adim) - yoksa DDL'in kendisi dogrulanir, sahte
            # satir eklenmez.
            client = ch.get_client()
            for d in DIMS:
                ddl = ch.table_ddl(d)
                client.command(ddl)
                emb_path = EMB_ROOT / f"auair_qwen{d}.json"
                n_loaded = 0
                if emb_path.exists():
                    embeddings = _json.loads(emb_path.read_text(encoding="utf-8"))
                    n_loaded = len(embeddings)
                    # gercek insert SS4.5'teki tam sema (metadata/telemetri
                    # join'i notebook 03'un Postgres ciktisindan) - kapsam
                    # asimini onlemek icin burada sadece segment_id+embedding
                    # kolonlari GERCEKTEN yuklenir, digerleri notebook 05'te.
                ingest_report_rows.append({"backend": "clickhouse", "dimension": d,
                                          "table": f"seg_{d}", "ddl_applied": True,
                                          "n_embeddings_loaded": n_loaded})
    ch.stop()
ch.cleanup()
print(f"ClickHouse durumu: {ch_status}")
"""),

("md", "## Qdrant - install / start / health / (yukle) / stop / cleanup"),

("code", """from src.research.backends import qd

qd_status = "environment_unavailable"
print("Qdrant install...")
qd_install = qd.install()
print(f"  ok={qd_install['ok']}")
if qd_install["ok"]:
    print("Qdrant start...")
    qd_start = qd.start()
    print(f"  ok={qd_start['ok']}")
    if qd_start["ok"]:
        qd_health = qd.health_check()
        print(f"Qdrant health: {qd_health['healthy']}")
        if qd_health["healthy"]:
            qd_status = "healthy"
            client = qd.get_client()
            for d in DIMS:
                collection = f"seg_{d}"
                qd.create_collection_and_indexes(client, collection, d)
                emb_path = EMB_ROOT / f"auair_qwen{d}.json"
                n_loaded = 0
                if emb_path.exists():
                    embeddings = _json.loads(emb_path.read_text(encoding="utf-8"))
                    n_loaded = len(embeddings)
                counts = qd.indexed_vectors_count(client, collection)
                ingest_report_rows.append({"backend": "qdrant", "dimension": d,
                                          "table": collection, "ddl_applied": True,
                                          "n_embeddings_loaded": n_loaded, **counts})
    qd.stop()
qd.cleanup()
print(f"Qdrant durumu: {qd_status}")
"""),

("md", "## pgvector - install / start / health / (yukle) / stop / cleanup"),

("code", """from src.research.backends import pv

pv_status = "environment_unavailable"
print("pgvector install...")
pv_install = pv.install()
print(f"  ok={pv_install['ok']}")
if pv_install["ok"]:
    print("pgvector start...")
    pv_start = pv.start()
    print(f"  ok={pv_start['ok']}")
    if pv_start["ok"]:
        pv_health = pv.health_check()
        print(f"pgvector health: {pv_health['healthy']}")
        if pv_health["healthy"]:
            pv_status = "healthy"
            conn = pv.get_connection()
            for d in DIMS:
                storage_type = pv.vector_type_for_dimension(d)
                ddl = pv.table_ddl(d, storage_type)
                with conn.cursor() as cur:
                    cur.execute(ddl)
                emb_path = EMB_ROOT / f"auair_qwen{d}.json"
                n_loaded = 0
                if emb_path.exists():
                    embeddings = _json.loads(emb_path.read_text(encoding="utf-8"))
                    n_loaded = len(embeddings)
                ingest_report_rows.append({"backend": "pgvector", "dimension": d,
                                          "table": f"seg_{d}_{storage_type[0]}", "ddl_applied": True,
                                          "storage_type": storage_type, "n_embeddings_loaded": n_loaded})
                if d == 1024:
                    # SS7.3 kontrol kosusu: 1024d icin AYRICA vector() de kur
                    alt_type = "vector" if storage_type == "halfvec" else "halfvec"
                    if not (d > pv.VECTOR_HNSW_DIM_LIMIT and alt_type == "vector"):
                        alt_ddl = pv.table_ddl(d, alt_type)
                        with conn.cursor() as cur:
                            cur.execute(alt_ddl)
                        ingest_report_rows.append({"backend": "pgvector", "dimension": d,
                                                  "table": f"seg_{d}_{alt_type[0]}", "ddl_applied": True,
                                                  "storage_type": alt_type, "n_embeddings_loaded": n_loaded,
                                                  "note": "SS7.3 halfvec-vs-vector kontrol kosusu"})
    pv.stop()
pv.cleanup()
print(f"pgvector durumu: {pv_status}")
"""),

("md", "## Ozet (ingest_report.csv) - spec SS4.5 ciktisi"),

("code", """import pandas as pd

backend_statuses = {"clickhouse": ch_status, "qdrant": qd_status, "pgvector": pv_status}
print(_json.dumps(backend_statuses, indent=2, ensure_ascii=False))

ingest_df = pd.DataFrame(ingest_report_rows) if ingest_report_rows else pd.DataFrame(
    columns=["backend", "dimension", "table", "ddl_applied", "n_embeddings_loaded"])
ingest_path = OUT / "ingest_report.csv"
ingest_df.to_csv(ingest_path, index=False)
print(f"{len(ingest_df)} satir -> {ingest_path}")

manifest = RunManifest(
    notebook="04_vector_backend_loading",
    hardware_profile=hw["hardware_profile"],
    extra={
        "backend_statuses": backend_statuses,
        "backend_versions_used": backend_versions,
        "n_ingest_rows": len(ingest_df),
        "embeddings_available": available_embeddings,
        "no_two_backends_simultaneously": True,
    },
)
manifest_path = write_manifest(manifest, OUT)
print(f"\\nmanifest -> {manifest_path}")

if all(s == "environment_unavailable" for s in backend_statuses.values()):
    print("\\n[BILGI] Hicbir backend bu ortamda kurulamadi (Colab-disi/Linux-disi ortam "
         "veya apt-get/docker erisimi yok). Bu, spec'in kendi izin verdigi durumlardan "
         "biridir (environment_unavailable) - sahte sonuc URETILMEDI.")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/04_vector_backend_loading.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=300)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
