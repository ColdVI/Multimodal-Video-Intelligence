"""Search-lab SQL katalogunu calistirip HTML + JSON kanit raporu uretir."""
from __future__ import annotations

import datetime as dt
import html
import json
import pathlib
import re
import time
import urllib.error
import urllib.request
from typing import Any

from search.sql_catalog import QUERY_SPECS, assert_read_only_sql, validate_catalog


FIXED_SETTINGS_SQL = """
SELECT name, value
FROM system.settings
WHERE name IN (
    'query_plan_try_use_vector_search',
    'vector_search_filter_strategy',
    'vector_search_with_rescoring',
    'vector_search_index_fetch_multiplier'
)
ORDER BY name
""".strip()


def _post(endpoint: str, sql: str, timeout_s: float = 30.0) -> str:
    request = urllib.request.Request(
        endpoint,
        data=sql.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ClickHouse HTTP {exc.code}: {body}") from exc


def _query_json(endpoint: str, sql: str) -> dict[str, Any]:
    assert_read_only_sql(sql)
    return json.loads(_post(endpoint, sql.rstrip() + "\nFORMAT JSON"))


def _explain(endpoint: str, sql: str) -> str:
    assert_read_only_sql(sql)
    return _post(endpoint, "EXPLAIN indexes = 1\n" + sql.rstrip() + "\nFORMAT TSVRaw")


_LONG_VECTOR_LITERAL = re.compile(
    r"\[(?:-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:,\s*)?){16,}\]"
)


def _compact_query_plan(plan: str) -> str:
    """EXPLAIN'in satira gomdugu 512d sorgu vektorunu okunabilir hale getir."""
    return _LONG_VECTOR_LITERAL.sub("[query_vector omitted]", plan)


def _ranking_signature(rows: list[dict]) -> list[tuple]:
    return [
        (row.get("video_id"), row.get("t_start"))
        for row in rows
    ]


def collect_clickhouse_evidence(endpoint: str = "http://localhost:8123/") -> dict:
    validate_catalog()
    version = _query_json(endpoint, "SELECT version() AS version")["data"][0]["version"]
    settings_rows = _query_json(endpoint, FIXED_SETTINGS_SQL)["data"]
    evidence = []
    for spec in QUERY_SPECS:
        sql = spec.sql()
        started = time.perf_counter()
        response = _query_json(endpoint, sql)
        client_ms = (time.perf_counter() - started) * 1000
        raw_plan = _explain(endpoint, sql) if spec.explain else ""
        evidence.append(
            {
                "query_id": spec.query_id,
                "title": spec.title,
                "kind": spec.kind,
                "description": spec.description,
                "filename": spec.filename,
                "sql": sql,
                "rows": response.get("data", []),
                "row_count": int(response.get("rows", len(response.get("data", [])))),
                "statistics": response.get("statistics", {}),
                "client_elapsed_ms": round(client_ms, 3),
                "vector_index_in_plan": "Description: vector_similarity" in raw_plan,
                "query_plan": _compact_query_plan(raw_plan),
            }
        )

    by_id = {row["query_id"]: row for row in evidence}
    exact_rows = by_id["similarity_exact_bruteforce"]["rows"]
    ann_rows = by_id["similarity_hnsw"]["rows"]
    vector_ranking_equal = _ranking_signature(exact_rows) == _ranking_signature(ann_rows)
    distance_deltas = [
        abs(float(exact.get("distance", 0.0)) - float(ann.get("distance", 0.0)))
        for exact, ann in zip(exact_rows, ann_rows)
        if (exact.get("video_id"), exact.get("t_start"))
        == (ann.get("video_id"), ann.get("t_start"))
    ]
    max_distance_delta = max(distance_deltas, default=0.0)
    table_rows = by_id["table_inventory"]["rows"]
    total_rows = sum(int(row.get("total_rows") or 0) for row in table_rows)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": endpoint,
        "clickhouse_version": version,
        "backend": "clickhouse",
        "catalog_query_count": len(evidence),
        "total_rows_across_model_tables": total_rows,
        "settings": {row["name"]: row["value"] for row in settings_rows},
        "vector_exact_and_hnsw_ranking_equal": vector_ranking_equal,
        "vector_exact_and_hnsw_distances_equal": max_distance_delta == 0.0,
        "vector_exact_and_hnsw_max_distance_delta": max_distance_delta,
        "queries": evidence,
        "methodology": [
            "Vector lab sorgulari tekrar edilebilir self-probe kullanir: query vector ilk kayitli video embedding'idir; dogal dil kalite benchmark'i degildir.",
            "Exact SQL filtresi saklanan kolon degerine kesindir; YOLO veya telemetri kolonunun gercek dunya dogrulugunu garanti etmez.",
            "Az satirli smoke tablolarinda HNSW hiz veya recall avantaji kanitlanamaz; brute-force zaten ucuzdur.",
            "HNSW sonucu yaklasik olabilir. Exact brute-force katalog sorgusu query_plan_try_use_vector_search=0 ile indeks optimizasyonunu kapatir.",
        ],
    }


def _table_html(rows: list[dict]) -> str:
    if not rows:
        return "<p>Sonuç yok.</p>"
    columns = list(rows[0])
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def render_clickhouse_report(evidence: dict) -> str:
    query_sections = []
    for item in evidence["queries"]:
        server_ms = float(item.get("statistics", {}).get("elapsed", 0.0)) * 1000
        plan = (
            f"<details><summary>EXPLAIN indexes = 1</summary><pre>{html.escape(item['query_plan'])}</pre></details>"
            if item.get("query_plan")
            else ""
        )
        query_sections.append(
            f"""
            <section class="query">
              <div class="badge">{html.escape(item['kind'])}</div>
              <h3>{html.escape(item['title'])}</h3>
              <p>{html.escape(item['description'])}</p>
              <div class="meta">Dosya: {html.escape(item['filename'])} · Satır: {item['row_count']} · Sunucu: {server_ms:.3f} ms · HTTP: {item['client_elapsed_ms']:.3f} ms · HNSW planı: {'kullanıldı' if item.get('vector_index_in_plan') else 'yok'}</div>
              <pre>{html.escape(item['sql'])}</pre>
              {_table_html(item['rows'])}
              {plan}
            </section>"""
        )
    ranking_equal = evidence["vector_exact_and_hnsw_ranking_equal"]
    distance_equal = evidence["vector_exact_and_hnsw_distances_equal"]
    ranking_text = "Aynı" if ranking_equal else "Farklı"
    distance_text = "Aynı" if distance_equal else "Farklı"
    methodology = "".join(
        f"<li>{html.escape(item)}</li>" for item in evidence.get("methodology", [])
    )
    settings = _table_html(
        [{"setting": key, "value": value} for key, value in evidence.get("settings", {}).items()]
    )
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ClickHouse Search Lab Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#172033;line-height:1.55;background:#f7f9fc}}
.hero{{background:linear-gradient(120deg,#102a56,#2458a6);color:white;border-radius:18px;padding:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:16px 0}}
.card,.query{{background:white;border:1px solid #dce4ef;border-radius:14px;padding:17px;margin:14px 0}}
.badge{{display:inline-block;background:#e6eeff;color:#174a9c;border-radius:999px;padding:4px 9px;font-size:12px}}
.meta{{color:#667085;font-size:13px;margin-bottom:10px}} pre{{background:#111827;color:#e5edf8;border-radius:10px;padding:13px;overflow:auto}}
.scroll{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left;white-space:nowrap}} th{{background:#f1f4f9}}
.warn{{border-left:5px solid #e7a500}}
</style></head><body>
<div class="hero"><h1>ClickHouse Search Lab</h1><p>Exact filtre · exact brute-force vector · HNSW · hybrid pre/postfilter</p></div>
<div class="grid">
  <div class="card"><b>ClickHouse</b><br>{html.escape(evidence['clickhouse_version'])}</div>
  <div class="card"><b>Katalog sorgusu</b><br>{evidence['catalog_query_count']}</div>
  <div class="card"><b>Toplam model satırı</b><br>{evidence['total_rows_across_model_tables']}</div>
  <div class="card"><b>Exact ↔ HNSW sırası</b><br>{ranking_text}</div>
  <div class="card"><b>Distance değerleri</b><br>{distance_text} · max Δ {evidence['vector_exact_and_hnsw_max_distance_delta']:.6f}</div>
</div>
<section class="card warn"><h2>Metodolojik sınırlar</h2><ul>{methodology}</ul></section>
<section class="card"><h2>Aktif vector ayarları</h2>{settings}</section>
<h2>Çalıştırılan SQL ve kanıtlar</h2>{''.join(query_sections)}
</body></html>"""


def write_clickhouse_report(
    repo_root: pathlib.Path,
    endpoint: str = "http://localhost:8123/",
) -> tuple[pathlib.Path, pathlib.Path]:
    repo_root = pathlib.Path(repo_root).resolve()
    output_dir = repo_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = collect_clickhouse_evidence(endpoint)
    json_path = output_dir / "clickhouse_search_report.json"
    html_path = output_dir / "clickhouse_search_report.html"
    json_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_clickhouse_report(evidence), encoding="utf-8")
    return html_path, json_path


__all__ = [
    "collect_clickhouse_evidence",
    "render_clickhouse_report",
    "write_clickhouse_report",
]
