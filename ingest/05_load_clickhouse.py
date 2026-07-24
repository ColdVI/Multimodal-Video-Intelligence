"""features.json + embeddings_<model>.json -> ClickHouse clips_<model> tablosu."""
import argparse
import json
import pathlib
import sys

import clickhouse_connect

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config


def split_sql_statements(sql):
    """Basit schema.sql dosyasini ClickHouse HTTP icin tekil ifadelere bol."""
    without_comments = "\n".join(
        line for line in sql.splitlines()
        if not line.lstrip().startswith("--")
    )
    return [statement.strip() for statement in without_comments.split(";")
            if statement.strip()]


def ensure_schema(client, schema_path="schema.sql"):
    sql = pathlib.Path(schema_path).read_text(encoding="utf-8")
    for statement in split_sql_statements(sql):
        client.command(statement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xclip_hf_zeroshot",
                     help="data/embeddings_<model>.json dosyasini yukler")
    args = ap.parse_args()

    cfg = load_config()
    ch = clickhouse_connect.get_client(
        host=cfg["clickhouse"]["host"], port=cfg["clickhouse"]["port"])
    ensure_schema(ch)

    emb_path = pathlib.Path(f"data/embeddings_{args.model}.json")
    if not emb_path.exists():
        print(f"HATA: {emb_path} yok. Once: python ingest/03_embed.py "
              f"--model {args.model}")
        raise SystemExit(1)

    features_path = pathlib.Path(cfg["paths"]["features"])
    if not features_path.exists():
        print(f"HATA: {features_path} yok. Once ingest/04_detect.py calistirin.")
        raise SystemExit(1)

    emb = {(e["video_id"], e["t_start"]): e["embedding"]
           for e in json.load(open(emb_path))}
    feats = json.load(open(features_path))

    rows = []
    skipped = 0
    for f in feats:
        key = (f["video_id"], f["t_start"])
        if key not in emb:
            skipped += 1
            continue
        rows.append([
            f["video_id"], f["t_start"], f["t_end"],
            f["person_count"], f["car_count"], f["bus_count"], f["truck_count"],
            f["is_night"], f["camera_motion"], f["brightness"],
            "visdrone", emb[key],
        ])

    table = f"clips_{args.model}"
    ch.insert(table, rows, column_names=[
        "video_id", "t_start", "t_end", "person_count", "car_count",
        "bus_count", "truck_count", "is_night", "camera_motion",
        "brightness", "platform", "embedding"])
    print(f"{len(rows)} satir {table} tablosuna yuklendi ({skipped} eslesmeyen atlandi)")


if __name__ == "__main__":
    main()
