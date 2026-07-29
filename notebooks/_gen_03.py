"""notebooks/03_postgres_metadata_telemetry.ipynb'i insa edip GERCEK
calistirarak uretir. Spec SS4.4. Embedding'e BAGIMLI DEGIL (notebook 02'nin
GPU kapisindan etkilenmez) - notebook 01'in GERCEK AU-AIR segment/telemetri
ciktisini gercek bir Postgres'e yukler.

NOT: bu depodaki ana POC docker-compose.yml KASITLI olarak Postgres
icermiyor (bkz. git log "Postgres on-arastirmasi: iliskisel butunluk
gerekmiyor, karar teyit edildi") - o karar POC'un KENDI mimarisi icindir.
Bu notebook AYRI, gecici, arastirma-amacli bir Postgres konteyneri kullanir
(research_postgres_faz6, port 5433) - ana docker-compose.yml'e DOKUNULMADI."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute

CELLS = [
("md", """# 03 - PostgreSQL metadata + telemetri

Spec SS4.4. Notebook 02'nin GPU kapisindan ETKILENMEZ - bu adim yalniz
notebook 01'in GERCEK AU-AIR segment/telemetri ciktisini yukler, embedding
gerektirmez. Gecici arastirma-amacli Postgres konteyneri: `research_postgres_faz6`
(port 5433) - ana `docker-compose.yml` DEGISTIRILMEDI."""),

("code", """import json
import pathlib
import sys

import pandas as pd
import psycopg

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research.config import DEFAULT as cfg
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest

OUT = cfg.research_root
hw = detect_hardware_profile()

DSN = "host=localhost port=5433 dbname=research user=postgres password=research"
conn = psycopg.connect(DSN, autocommit=True)
print("Postgres baglantisi kuruldu:", conn.info.server_version)
"""),

("md", "## Sema (spec SS4.4 - minimal sema, genisletilmedi)"),

("code", """SCHEMA_SQL = '''
DROP TABLE IF EXISTS segment_telemetry, segment_metadata, segments, videos, datasets CASCADE;

CREATE TABLE datasets (
    dataset_id      text PRIMARY KEY,
    dataset_version text NOT NULL,
    source_hash     text NOT NULL
);

CREATE TABLE videos (
    dataset_id text NOT NULL,
    video_id   text NOT NULL,
    source_uri text,
    split      text,
    duration_s double precision,
    PRIMARY KEY (dataset_id, video_id)
);

CREATE TABLE segments (
    segment_id text PRIMARY KEY,
    dataset_id text NOT NULL,
    video_id   text NOT NULL,
    t_start    double precision NOT NULL,
    t_end      double precision NOT NULL
);

CREATE TABLE segment_metadata (
    segment_id    text PRIMARY KEY REFERENCES segments(segment_id),
    person_count  integer,
    vehicle_count integer,
    bus_count     integer,
    object_classes text[],
    brightness    real,
    camera_motion real
);

CREATE TABLE segment_telemetry (
    segment_id      text PRIMARY KEY REFERENCES segments(segment_id),
    timestamp_start timestamptz,
    timestamp_end   timestamptz,
    latitude        double precision,
    longitude       double precision,
    altitude_m      real,
    velocity_mps    real,
    roll            real,
    pitch           real,
    yaw             real,
    yaw_rate        real,
    imu_summary     jsonb
);

CREATE INDEX ON segment_metadata (person_count);
CREATE INDEX ON segment_metadata (vehicle_count);
CREATE INDEX ON segment_telemetry (altitude_m);
CREATE INDEX ON segment_telemetry (velocity_mps);
'''

with conn.cursor() as cur:
    cur.execute(SCHEMA_SQL)
print("sema kuruldu (5 tablo + 4 index).")
"""),

("md", "## Yukleme (COPY) - notebook 01'in GERCEK AU-AIR ciktisi"),

("code", """seg_df = pd.read_parquet(OUT / "auair_segments.parquet")
tel_df = pd.read_parquet(OUT / "auair_telemetry.parquet")
manifest_01 = json.loads((OUT / "01_auair_download_and_validation_manifest.json").read_text(encoding="utf-8"))

with conn.cursor() as cur:
    cur.execute("INSERT INTO datasets VALUES (%s, %s, %s)",
               ("auair", "1.0-2019", manifest_01["extra"]["annotations_sha256"]))

    video_ids = seg_df["video_id"].unique()
    for vid in video_ids:
        cur.execute("INSERT INTO videos (dataset_id, video_id, split) VALUES (%s,%s,%s)",
                   ("auair", vid, "n/a"))

    for _, row in seg_df.iterrows():
        cur.execute("INSERT INTO segments VALUES (%s,%s,%s,%s,%s)",
                   (row["segment_id"], "auair", row["video_id"], row["t_start"], row["t_end"]))

    tel_by_seg = tel_df.set_index("segment_id")
    for seg_id, t in tel_by_seg.iterrows():
        cur.execute(\"\"\"INSERT INTO segment_metadata (segment_id, person_count, vehicle_count)
                      VALUES (%s,%s,%s)\"\"\",
                   (seg_id, int(t["person_count"]), int(t["vehicle_count"])))
        cur.execute(\"\"\"INSERT INTO segment_telemetry
                      (segment_id, altitude_m, velocity_mps, roll, pitch, yaw, yaw_rate)
                      VALUES (%s,%s,%s,%s,%s,%s,%s)\"\"\",
                   (seg_id, float(t["altitude_m"]), float(t["velocity_mps"]),
                    float(t["roll"]), float(t["pitch"]), float(t["yaw"]), float(t["yaw_rate"])))

with conn.cursor() as cur:
    counts = {}
    for table in ("datasets", "videos", "segments", "segment_metadata", "segment_telemetry"):
        cur.execute(f"SELECT count(*) FROM {table}")
        counts[table] = cur.fetchone()[0]
print(json.dumps(counts, indent=2))
assert counts["segments"] == len(seg_df), "satir sayisi dogrulamasi basarisiz"
assert counts["segment_telemetry"] == len(tel_df), "satir sayisi dogrulamasi basarisiz"
print("satir sayisi dogrulamasi GECTI (kaynak parquet == Postgres tablo satiri).")
"""),

("md", "## Secicilik dogrulamasi (spec SS4.4 - her hedef seviyede GERCEKTE kac segment donuyor)"),

("code", """selectivity_thresholds = json.loads((OUT / "selectivity_thresholds.json").read_text(encoding="utf-8"))

verification_rows = []
with conn.cursor() as cur:
    for field, direction, column in [
        ("altitude_m", "less_than", "altitude_m"),
        ("velocity_mps", "greater_than", "velocity_mps"),
        ("person_count", "greater_than", "person_count"),
        ("vehicle_count", "greater_than", "vehicle_count"),
    ]:
        levels = selectivity_thresholds[field]
        table = "segment_telemetry" if column in ("altitude_m", "velocity_mps") else "segment_metadata"
        op = "<" if direction == "less_than" else ">"
        for p, info in levels.items():
            theta = info["threshold"]
            if theta is None:
                continue
            cur.execute(f"SELECT count(*) FROM {table} WHERE {column} {op} %s", (theta,))
            n_returned = cur.fetchone()[0]
            verification_rows.append({
                "field": field, "target_p": p, "threshold": theta,
                "n_returned_from_postgres": n_returned,
                "actual_selectivity_from_postgres": n_returned / counts["segments"],
                "actual_selectivity_from_derive_thresholds": info["actual_selectivity"],
            })

ver_df = pd.DataFrame(verification_rows)
ver_path = OUT / "auair_selectivity_postgres_verification.csv"
ver_df.to_csv(ver_path, index=False)
print(ver_df.to_string(index=False))
print(f"\\n-> {ver_path}")
"""),

("md", "## Ozet"),

("code", """pg_report_path = OUT / "pg_load_report.md"
pg_report = f'''# PostgreSQL yukleme raporu (GERCEK calistirmadan)

Konteyner: `research_postgres_faz6` (gecici, port 5433 - ana docker-compose.yml
DEGISTIRILMEDI, ana POC'un "Postgres yok" karari BOZULMADI).

## Satir sayilari (kaynak parquet ile birebir dogrulandi)
{json.dumps(counts, indent=2, ensure_ascii=False)}

## Secicilik dogrulamasi
`derive_thresholds()` (numpy quantile, notebook 01) ile canli Postgres
sorgusunun (`WHERE column < / > esik`) dondurdugu GERCEK satir sayisi
karsilastirildi - {len(ver_df)} satir, tam liste `auair_selectivity_postgres_verification.csv`'de.
Iki kaynak arasindaki fark yalniz kayan-nokta/quantile enterpolasyon
farklarindan kaynaklanabilir (n=8 video x pencere sayisi kucuk oldugu icin
p=0.001 gibi asiri uc seviyelerde 1-2 satirlik sapma beklenir).

## Kapsam disi
Bu notebook `segment_metadata.object_classes/brightness/camera_motion`
kolonlarini DOLDURMADI - bunlar AU-AIR annotation'inin bbox/class alanindan
turetilebilir ama spec'in bu notebook icin istedigi minimal kapsam
(person_count, vehicle_count, telemetri) ile sinirli tutuldu.
'''
pg_report_path.write_text(pg_report, encoding="utf-8")
print(pg_report)

conn.close()

manifest = RunManifest(
    notebook="03_postgres_metadata_telemetry",
    hardware_profile=hw["hardware_profile"],
    dataset_id="auair",
    extra={"row_counts": counts, "postgres_container": "research_postgres_faz6:5433"},
)
manifest_path = write_manifest(manifest, OUT)
print(f"\\nmanifest -> {manifest_path}")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/03_postgres_metadata_telemetry.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=300)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
