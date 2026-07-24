"""Hibrit ClickHouse sorgusu: filtre kolonlari + vektor benzerligi.
Model basina ayri tablo (bkz. schema.sql) - tablo adi clips_<model_name>."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config
from models import get_embedder

from .parser import parse

_client = None


def _get_client(cfg):
    global _client
    if _client is None:
        import clickhouse_connect
        _client = clickhouse_connect.get_client(
            host=cfg["clickhouse"]["host"], port=cfg["clickhouse"]["port"])
    return _client


def _fmt_vector(vec) -> str:
    # ClickHouse array literal. Sabit ondalik format kullaniyoruz cunku
    # Python'un varsayilan float repr'i bilimsel gosterime (1e-05) dusebilir,
    # bu ClickHouse array literal ayristirmasinda soruna yol acabilir.
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def search(q: str, model_name: str, top_k: int = 200,
           use_filters: bool = True):
    """Donus: [(video_id, t_start, t_end, dist), ...] dist artan sirada."""
    cfg = load_config()
    ch = _get_client(cfg)
    p = parse(q)
    emb = get_embedder(model_name)
    qvec = emb.embed_text(p.semantic)

    where = "1"
    if use_filters and p.filters:
        where = " AND ".join(f"{c} {op} {v}" for c, op, v in p.filters)

    table = f"clips_{model_name}"
    sql = f"""
        SELECT video_id, t_start, t_end,
               cosineDistance(embedding, {_fmt_vector(qvec)}) AS dist
        FROM {table}
        WHERE {where}
        ORDER BY dist ASC
        LIMIT {top_k}
    """
    return ch.query(sql).result_rows
