"""pipeline_validation.html'in v2'si: MSR-VTT 1000/1000 gerçek video/sorgu
kalite benchmarkı olduğunu açık bir rozetle çerçeveler (X-CLIP checkpoint
soyu ve baseline yön düzeltmesi zaten JSON'da var - bkz. scripts/
validate_msrvtt.py). Eski pipeline_validation.html DOKUNULMAZ."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reports.msrvtt_validation_html import render_msrvtt_report
from reports.scope_badge import render_scope_badge

ARTIFACTS = pathlib.Path("artifacts")


def main():
    evidence = json.loads((ARTIFACTS / "pipeline_validation.json").read_text(encoding="utf-8"))
    measured = next(iter(evidence["results"].values()))["measured"]
    badge = render_scope_badge(
        kind="REAL",
        dataset="MSR-VTT 1k-A test split (standart yayınlanmış benchmark)",
        count=measured["n_pairs_evaluated"], count_label="video/caption çifti (tam split, alt küme değil)",
        purpose=("xclip_hf_zeroshot checkpoint'inin gerçek text-to-video retrieval kalitesini, "
                 "yayınlanmış bir literatür baseline'ına karşı ölçmek - bu depodaki TEK gerçek "
                 "model-kalite benchmarkı (diğer raporlar gecikme/plan ölçer, kalite değil)."),
        can_claim=[
            "R@1/R@5/R@10/MedR/MeanR - standart 1k-A protokolüyle karşılaştırılabilir",
            "MeanR'nin rastgele şanstan kaç kat iyi olduğu (baseline tartışmasından bağımsız)",
            "Zero-shot CLIP-straight baseline'ına karşı hangi metriklerin bayraklı kaldığı",
        ],
        cannot_claim=[
            "VisDrone'daki arama kalitesi (farklı domain/görev - drone görüntüsü vs. genel video, "
            "windowed interval retrieval vs. whole-clip T2V)",
            "Ma ve ark. (retrieval-fine-tuned) X-CLIP'in sonucu (entegre edilmedi, bkz. script notu)",
        ],
    )
    html_out = render_msrvtt_report(evidence, scope_badge_html=badge)
    out_path = ARTIFACTS / "pipeline_validation_v2.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"v2 HTML rapor: {out_path}")
    print(f"  (eski {ARTIFACTS / 'pipeline_validation.html'} DOKUNULMADI)")


if __name__ == "__main__":
    main()
