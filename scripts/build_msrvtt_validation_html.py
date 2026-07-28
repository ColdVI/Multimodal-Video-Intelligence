"""artifacts/pipeline_validation.json'i artifacts/pipeline_validation.html'e render eder."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reports.msrvtt_validation_html import render_msrvtt_report

ARTIFACTS = pathlib.Path("artifacts")


def main():
    evidence = json.loads((ARTIFACTS / "pipeline_validation.json").read_text(encoding="utf-8"))
    html = render_msrvtt_report(evidence)
    out_path = ARTIFACTS / "pipeline_validation.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML rapor: {out_path}")


if __name__ == "__main__":
    main()
