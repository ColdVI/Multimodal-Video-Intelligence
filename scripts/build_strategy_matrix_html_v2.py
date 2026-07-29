"""strategy_matrix_report.html'in v2'si: 73 gerçek pencere ile 100K
sentetik satırı AYRI, açık rozetlerle etiketler. Aynı frozen JSON
kanıtlarını (yeniden ölçmeden) kullanır - scripts/build_strategy_matrix_
html.py ile aynı veri, farklı render. Eski .html DOKUNULMAZ."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reports.scope_badge import render_scope_badge
from reports.strategy_matrix_html import render_strategy_report

ARTIFACTS = pathlib.Path("artifacts")


def main():
    small_scale = json.loads((ARTIFACTS / "strategy_matrix_report.json").read_text(encoding="utf-8"))

    scale_path = ARTIFACTS / "scale_evidence_bench_scale_512.json"
    scale_100k = json.loads(scale_path.read_text(encoding="utf-8")) if scale_path.exists() else None

    build_path = ARTIFACTS / "scale_table_build.json"
    memory_projection = None
    scale_corpus_size = None
    if build_path.exists():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        scale_corpus_size = build["n_rows"]
        per_row_bytes = build["indices"][0]["size_bytes"] / build["n_rows"]
        memory_projection = {
            "1M satır 512d (ölçülen)": round(per_row_bytes * 1_000_000 / 1e9, 2),
            "10M satır 512d (ölçülen)": round(per_row_bytes * 10_000_000 / 1e9, 2),
            "1M satır 1152d (boyut oranıyla ekstrapole)": round(
                per_row_bytes * (1152 / 512) * 1_000_000 / 1e9, 2),
            "10M satır 1152d (boyut oranıyla ekstrapole)": round(
                per_row_bytes * (1152 / 512) * 10_000_000 / 1e9, 2),
        }

    real_badge = render_scope_badge(
        kind="REAL",
        dataset="VisDrone bench subset (19 sekans, 73 pencere, gerçek X-CLIP/SigLIP2 embedding)",
        count=73, count_label="pencere (gerçek video, gerçek YOLO+embedding)",
        purpose=("4 ClickHouse arama stratejisinin (bruteforce/HNSW/prefilter/postfilter) "
                 "gerçek veri üzerinde gecikme ve query_plan davranışını karşılaştırmak."),
        can_claim=[
            "Bu 73 satırda hangi stratejinin en hızlı olduğu (p50/p95 gecikme)",
            "'auto' (postfiltering) stratejisinin seçici filtrede 0 satır dönebildiği (gerçek bulgu)",
            "fetch_multiplier/ef_search ayarlarının bu ölçekte gecikmeye etkisi",
        ],
        cannot_claim=[
            "100K/1M/1PB ölçekte aynı stratejinin en hızlı kalacağı - bkz. aşağıdaki SENTETİK bölüm",
            "Model kalitesi/retrieval doğruluğu (bu rapor yalnız gecikme/plan ölçer)",
        ],
    )
    synthetic_badge = render_scope_badge(
        kind="SYNTHETIC",
        dataset="bench_scale_512 (kontrollü gürültüyle replike edilmiş sentetik vektörler)",
        count=scale_corpus_size or 100_000, count_label="satır (gerçek embedding DEĞİL)",
        purpose=("HNSW index inşa süresi/boyutu ve sorgu gecikmesinin satır sayısıyla nasıl "
                 "ölçeklendiğini görmek - gerçek video/embedding kalitesiyle İLGİSİZ."),
        can_claim=[
            "Index boyutunun satır sayısıyla nasıl büyüdüğü (ölçülen, ekstrapole edilebilir)",
            "Bu ölçekte sorgu gecikmesi",
        ],
        cannot_claim=[
            "Retrieval KALİTESİ (vektörler sentetik, gerçek bir sorguya karşılık gelmiyor)",
            "1 PB (~419.430.400 pencere) davranışı - bu satır sayısından ~4194x uzak, "
            "ekstrapolasyondur, ölçüm değil",
        ],
    )

    html_out = render_strategy_report(
        small_scale, scale_100k=scale_100k, memory_projection=memory_projection,
        scale_corpus_size=scale_corpus_size, real_badge_html=real_badge,
        synthetic_badge_html=synthetic_badge)
    out_path = ARTIFACTS / "strategy_matrix_report_v2.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"v2 HTML rapor: {out_path}")
    print(f"  (eski {ARTIFACTS / 'strategy_matrix_report.html'} DOKUNULMADI)")


if __name__ == "__main__":
    main()
