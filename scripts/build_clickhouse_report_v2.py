"""artifacts/clickhouse_search_report.json'daki (2026-07-24, DONDURULMUS
kanit - 14 satir/2 tablo, "7 satirlik" smoke durumu) kaniti YENIDEN
TOPLAMADAN v2 (kapsam rozetli) HTML'e render eder. Canli ClickHouse
GEREKTIRMEZ - o tarihteki durumu yansitir, SIMDIKI (bu depoda artik 73+
satir) durumu DEGIL; bu bilerek boyle, "7 satirlik smoke raporu" neyse o
kaldi. Eski artifacts/clickhouse_search_report.html DOKUNULMAZ."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reports.clickhouse_search_report import render_clickhouse_report
from reports.scope_badge import render_scope_badge

ARTIFACTS = pathlib.Path("artifacts")


def main():
    evidence = json.loads((ARTIFACTS / "clickhouse_search_report.json").read_text(encoding="utf-8"))
    badge = render_scope_badge(
        kind="SMOKE_TEST",
        dataset=(f"VisDrone 5-sekans smoke ({evidence['total_rows_across_model_tables']} "
                "satır, 2 tablo)"),
        count=evidence["catalog_query_count"], count_label="katalog sorgusu",
        purpose=("ClickHouse hibrit arama SQL'inin (exact filtre, exact brute-force vector, "
                 "HNSW, iki hibrit strateji) sözdizimsel olarak çalıştığını ve query_plan'in "
                 "beklenen indeksi kullandığını doğrulamak - bir HIZ veya ÖLÇEK testi DEĞİL."),
        can_claim=[
            "7 SQL sorgusunun hepsi hatasız çalışıyor",
            "EXPLAIN indexes=1 planında HNSW indeksi beklenen sorgularda görünüyor",
            "Bu 14 satırlık veride exact ve HNSW aynı sıralamayı üretiyor",
        ],
        cannot_claim=[
            "Gerçek ölçekte (73, 100K, 1M+) gecikme veya HNSW recall davranışı - "
            "bkz. strategy_matrix_report_v2.html",
            "Herhangi bir modelin arama KALİTESİ (sorgu vektörü kaydedilmiş ilk "
            "embedding'dir, doğal dil kalite testi değildir)",
        ],
        generated_at=evidence.get("generated_at_utc"),
    )
    html_out = render_clickhouse_report(evidence, scope_badge_html=badge)
    out_path = ARTIFACTS / "clickhouse_search_report_v2.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"v2 HTML rapor: {out_path}")
    print(f"  (eski {ARTIFACTS / 'clickhouse_search_report.html'} DOKUNULMADI)")


if __name__ == "__main__":
    main()
