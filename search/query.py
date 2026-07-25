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


# Faz 2 strateji matrisi (sql/search_lab/08-15) ile deger olarak senkron:
# 'auto' mevcut davranis (SETTINGS eklenmez - bu ClickHouse surumunde
# varsayilan 'auto' postfiltering'e esittir), digerleri o dosyalardaki
# SETTINGS satirlarinin birebir ayni.
_STRATEGY_SETTINGS = {
    "auto": "",
    "exact": "SETTINGS query_plan_try_use_vector_search = 0",
    "prefilter": "SETTINGS vector_search_filter_strategy = 'prefilter'",
    "postfilter_rescore": (
        "SETTINGS vector_search_filter_strategy = 'postfilter', "
        "vector_search_with_rescoring = 1, vector_search_index_fetch_multiplier = 5"
    ),
}


def _build_sql(where: str, table: str, qvec, top_k: int, strategy: str) -> str:
    if strategy not in _STRATEGY_SETTINGS:
        raise ValueError(
            f"bilinmeyen strategy: {strategy!r}. Gecerli: {sorted(_STRATEGY_SETTINGS)}")
    return f"""
        SELECT video_id, t_start, t_end,
               cosineDistance(embedding, {_fmt_vector(qvec)}) AS dist
        FROM {table}
        WHERE {where}
        ORDER BY dist ASC
        LIMIT {top_k}
        {_STRATEGY_SETTINGS[strategy]}
    """


def search(q: str, model_name: str, top_k: int = 200,
           use_filters: bool = True, strategy: str = "auto"):
    """Donus: [(video_id, t_start, t_end, dist), ...] dist artan sirada.
    strategy: 'auto' (varsayilan, onceki davranisla birebir ayni) | 'exact' |
    'prefilter' | 'postfilter_rescore' (bkz. _STRATEGY_SETTINGS)."""
    cfg = load_config()
    ch = _get_client(cfg)
    p = parse(q)
    emb = get_embedder(model_name)
    qvec = emb.embed_text(p.semantic)

    where = "1"
    if use_filters and p.filters:
        where = " AND ".join(f"{c} {op} {v}" for c, op, v in p.filters)

    table = f"clips_{model_name}"
    sql = _build_sql(where, table, qvec, top_k, strategy)
    return ch.query(sql).result_rows
