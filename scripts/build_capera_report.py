"""artifacts/capera_validation.json'i artifacts/capera_validation.html'e
render eder (scripts/migrate_capera_results.py'nin ciktisini)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reports.capera_validation_html import render_capera_report
from reports.scope_badge import render_scope_badge

ARTIFACTS = pathlib.Path("artifacts")


def main():
    path = ARTIFACTS / "capera_validation.json"
    if not path.exists():
        print(f"HATA: {path} yok - once scripts/migrate_capera_results.py calistirin.")
        raise SystemExit(1)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    dm = evidence["dataset_manifest"]
    counts = evidence.get("counts", {})
    eval_videos = counts.get("evaluated_video_count", "unknown")
    eval_queries = counts.get("evaluated_query_count", "unknown")
    eval_failed = counts.get("failed_video_count", "unknown")

    badge = render_scope_badge(
        kind="REAL",
        dataset=(f"CapERA (ERA aerial event captioning) - manifest {dm['item_count']} video, "
                f"değerlendirilen {eval_videos} video"),
        count=eval_queries, count_label="değerlendirilmiş sorgu (all_results.json - "
                                        f"manifest'te {dm['query_count']} sorgu var, tamamı değil)",
        purpose=("3 embedding modelinin (Qwen3-VL-Embedding-2B/8B, VideoCLIP-XL) CapERA "
                 "üzerinde zero-shot T2V retrieval kalitesini karşılaştırmak - kullanıcı "
                 "tarafından Google Drive'dan kopyalanan, önceden üretilmiş GERÇEK sonuçlar "
                 "(bu depoda yeniden üretilmedi/koşulmadı)."),
        can_claim=[
            f"3 modelin R@1/R@5/R@10/MRR karşılaştırması (aynı {eval_videos} video/"
            f"{eval_queries} sorguya karşı, {eval_failed} başarısız)",
            "Qwen3-VL-Embedding-8B'nin bu 3 model içinde en yüksek recall'a sahip olduğu "
            "(2B'ye göre küçük fark, VideoCLIP-XL'e göre belirgin fark)",
            "Model başına GPU bellek/gecikme karşılaştırması",
        ],
        cannot_claim=[
            "Per-kategori veya per-sorgu kırılım (o dosyalar bu depoya kopyalanmadı)",
            "ebind-audio-vision sonucu (notebook'ta adaptörü var ama sonuç yok)",
            "VisDrone veya MSR-VTT ile doğrudan karşılaştırma (farklı dataset/protokol/model seti)",
            "Bu depoda yeniden üretilebilirlik (video dosyaları yerel değil, GPU yok)",
        ],
    )
    html_out = render_capera_report(evidence, scope_badge_html=badge)
    out_path = ARTIFACTS / "capera_validation.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"HTML rapor: {out_path}")


if __name__ == "__main__":
    main()
