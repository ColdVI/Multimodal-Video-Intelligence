"""ClickHouse search-lab SQL katalogu.

SQL dosyalari insanlarin ClickHouse /play ekraninda kopyalayabilmesi icin
ayri tutulur. Test ve rapor kodu da ayni dosyalari okur; boylece dokumandaki
sorgu ile gercekte kosulan sorgu birbirinden kopmaz.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SQL_ROOT = REPO_ROOT / "sql" / "search_lab"


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    title: str
    kind: str
    filename: str
    description: str
    explain: bool = True

    @property
    def path(self) -> pathlib.Path:
        return SQL_ROOT / self.filename

    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8").strip().rstrip(";")


QUERY_SPECS = (
    QuerySpec(
        "table_inventory",
        "Tablo envanteri",
        "inventory",
        "01_table_inventory.sql",
        "Model tablolarinin satir ve disk boyutlarini listeler.",
        explain=False,
    ),
    QuerySpec(
        "index_inventory",
        "Indeks envanteri",
        "inventory",
        "02_index_inventory.sql",
        "Minmax ve HNSW vector_similarity indekslerini listeler.",
        explain=False,
    ),
    QuerySpec(
        "exact_filter",
        "Exact yapilandirilmis filtre",
        "exact_filter",
        "03_exact_filter.sql",
        "Embedding kullanmadan bus_count ve person_count kolonlarini deterministik filtreler.",
    ),
    QuerySpec(
        "similarity_exact_bruteforce",
        "Exact brute-force vector search",
        "vector_exact",
        "04_similarity_exact_bruteforce.sql",
        "HNSW optimizasyonunu kapatip tum adaylarin cosine distance degerini hesaplar.",
    ),
    QuerySpec(
        "similarity_hnsw",
        "HNSW approximate similarity search",
        "vector_ann",
        "05_similarity_hnsw.sql",
        "ClickHouse query planner'in HNSW vector_similarity indeksini kullanmasina izin verir.",
    ),
    QuerySpec(
        "hybrid_prefilter",
        "Hybrid prefilter",
        "hybrid_prefilter",
        "06_hybrid_prefilter.sql",
        "Exact kolon filtrelerini once uygular, kalan adaylarda brute-force vector search yapar.",
    ),
    QuerySpec(
        "hybrid_postfilter_rescore",
        "Hybrid postfilter + rescore",
        "hybrid_postfilter",
        "07_hybrid_postfilter_rescore.sql",
        "HNSW adaylarini bulur, filtreler ve full-precision vektorlerle yeniden skorlar.",
    ),
)


def get_query_spec(query_id: str) -> QuerySpec:
    for spec in QUERY_SPECS:
        if spec.query_id == query_id:
            return spec
    raise KeyError(f"bilinmeyen query_id: {query_id}")


def assert_read_only_sql(sql: str) -> None:
    """Rapor runner'inin yalnizca read-only lab SQL'i calistirmasini sagla."""
    normalized = re.sub(r"--.*?$|/\*.*?\*/", " ", sql, flags=re.M | re.S).strip()
    if not re.match(r"^(SELECT|WITH)\b", normalized, flags=re.I):
        raise ValueError("search-lab SQL SELECT veya WITH ile baslamali")
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|RENAME|OPTIMIZE|KILL)\b",
        flags=re.I,
    )
    match = forbidden.search(normalized)
    if match:
        raise ValueError(f"read-only katalogda yasak SQL: {match.group(1)}")
    if ";" in normalized:
        raise ValueError("search-lab dosyasi tek statement olmali; noktalivirgulu kaldirin")


def validate_catalog() -> None:
    ids = [spec.query_id for spec in QUERY_SPECS]
    files = [spec.filename for spec in QUERY_SPECS]
    if len(ids) != len(set(ids)):
        raise ValueError("query_id degerleri benzersiz olmali")
    if len(files) != len(set(files)):
        raise ValueError("SQL dosyalari benzersiz olmali")
    for spec in QUERY_SPECS:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        assert_read_only_sql(spec.sql())


__all__ = [
    "QUERY_SPECS",
    "QuerySpec",
    "assert_read_only_sql",
    "get_query_spec",
    "validate_catalog",
]
