"""artifacts/strategy_matrix_report.json + scale_evidence_*.json + memory
projection'i tek strategy_matrix_report.html'de birlestirir."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reports.strategy_matrix_html import render_strategy_report

ARTIFACTS = pathlib.Path("artifacts")


def main():
    small_scale = json.loads((ARTIFACTS / "strategy_matrix_report.json").read_text(encoding="utf-8"))

    scale_path = ARTIFACTS / "scale_evidence_bench_scale_512.json"
    scale_100k = json.loads(scale_path.read_text(encoding="utf-8")) if scale_path.exists() else None

    build_path = ARTIFACTS / "scale_table_build.json"
    memory_projection = None
    if build_path.exists():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        per_row_bytes = build["indices"][0]["size_bytes"] / build["n_rows"]
        memory_projection = {
            "1M satır 512d (ölçülen)": round(per_row_bytes * 1_000_000 / 1e9, 2),
            "10M satır 512d (ölçülen)": round(per_row_bytes * 10_000_000 / 1e9, 2),
            "1M satır 1152d (boyut oranıyla ekstrapole)": round(
                per_row_bytes * (1152 / 512) * 1_000_000 / 1e9, 2),
            "10M satır 1152d (boyut oranıyla ekstrapole)": round(
                per_row_bytes * (1152 / 512) * 10_000_000 / 1e9, 2),
        }

    html = render_strategy_report(small_scale, scale_100k=scale_100k,
                                  memory_projection=memory_projection)
    out_path = ARTIFACTS / "strategy_matrix_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML rapor: {out_path}")


if __name__ == "__main__":
    main()
