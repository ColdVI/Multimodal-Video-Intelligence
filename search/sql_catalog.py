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
    # Faz 2 strateji matrisi icin: dosyadaki dort stratejiden (bruteforce/
    # hnsw/prefilter/postfilter) hangisi ve hangi filtre secicilikte
    # (loose: person_count>=1, strict: bus_count>=1 AND person_count>=3).
    # Orijinal 7 smoke sorgusunda None kalir.
    strategy: str = None
    selectivity: str = None

    @property
    def path(self) -> pathlib.Path:
        return SQL_ROOT / self.filename

    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8").strip().rstrip(";")

    def sql_for_table(self, table: str) -> str:
        """Faz 2'nin 'iki tablo' ekseni: dosyanin kendisi (insan/test'in
        okudugu) tek tabloyu (clips_xclip_hf_zeroshot) hedefler; ikinci
        tablo icin ayni sorgu seklini runtime'da tablo adini degistirerek
        uretiriz - mekanik bir isim degisimi, sorgu mantigi aynidir."""
        return self.sql().replace("clips_xclip_hf_zeroshot", table)


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
    # Faz 2 strateji matrisi: dort strateji x iki filtre secicilik (loose:
    # person_count>=1, strict: bus_count>=1 AND person_count>=3). Ikinci
    # tablo (clips_siglip2_frameavg) rapor tarafinda sql_for_table() ile
    # ayni dosyadan uretilir - bkz. docs/codex/05_..._PLANI.md Faz 2 madde 1.
    QuerySpec(
        "matrix_bruteforce_loose", "Matrix: brute-force x loose", "matrix",
        "08_matrix_bruteforce_loose.sql",
        "Exact brute-force vector search + person_count>=1.",
        strategy="bruteforce", selectivity="loose",
    ),
    QuerySpec(
        "matrix_bruteforce_strict", "Matrix: brute-force x strict", "matrix",
        "09_matrix_bruteforce_strict.sql",
        "Exact brute-force vector search + bus_count>=1 AND person_count>=3.",
        strategy="bruteforce", selectivity="strict",
    ),
    QuerySpec(
        "matrix_hnsw_loose", "Matrix: HNSW x loose", "matrix",
        "10_matrix_hnsw_loose.sql",
        "HNSW (varsayilan vector_search_filter_strategy='auto') + person_count>=1.",
        strategy="hnsw", selectivity="loose",
    ),
    QuerySpec(
        "matrix_hnsw_strict", "Matrix: HNSW x strict", "matrix",
        "11_matrix_hnsw_strict.sql",
        "HNSW (varsayilan) + bus_count>=1 AND person_count>=3.",
        strategy="hnsw", selectivity="strict",
    ),
    QuerySpec(
        "matrix_prefilter_loose", "Matrix: prefilter x loose", "matrix",
        "12_matrix_prefilter_loose.sql",
        "vector_search_filter_strategy='prefilter' + person_count>=1.",
        strategy="prefilter", selectivity="loose",
    ),
    QuerySpec(
        "matrix_prefilter_strict", "Matrix: prefilter x strict", "matrix",
        "13_matrix_prefilter_strict.sql",
        "vector_search_filter_strategy='prefilter' + bus_count>=1 AND person_count>=3.",
        strategy="prefilter", selectivity="strict",
    ),
    QuerySpec(
        "matrix_postfilter_loose", "Matrix: postfilter+rescore x loose", "matrix",
        "14_matrix_postfilter_loose.sql",
        "postfilter + rescoring + fetch_multiplier=5 + person_count>=1.",
        strategy="postfilter_rescore", selectivity="loose",
    ),
    QuerySpec(
        "matrix_postfilter_strict", "Matrix: postfilter+rescore x strict", "matrix",
        "15_matrix_postfilter_strict.sql",
        "postfilter + rescoring + fetch_multiplier=5 + bus_count>=1 AND person_count>=3.",
        strategy="postfilter_rescore", selectivity="strict",
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
