from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "faz11"


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1440, 900), "#0b1020")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=18)
    draw.rounded_rectangle((100, 150, 1340, 750), radius=24, fill="#121a2f", outline="#52617f", width=2)
    draw.text((150, 220), "FAZ 11 UI / MEDIA ACCEPTANCE - NOT RUN", font=font, fill="#f6c453")
    lines = [
        "Implementation and pure contract tests are complete.",
        "A live Docker API/UI stack, active institutional run, local MP4 and browser runtime",
        "were unavailable in this environment; no successful screenshot is being claimed.",
        "Run the required command recorded in ui_smoke.json on the target NVIDIA Linux host.",
    ]
    for index, line in enumerate(lines):
        draw.text((150, 315 + index * 55), line, font=small, fill="#d7deef")
    draw.text((150, 650), datetime.now(timezone.utc).isoformat(), font=small, fill="#8d9ab8")
    image.save(ARTIFACT_ROOT / "ui_smoke.png", format="PNG")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_run",
        "reason": (
            "Docker daemon, active institutional run/local MP4, and the required in-app browser Node REPL "
            "runtime were unavailable; the PNG is explicitly labelled NOT RUN and is not a success screenshot."
        ),
        "implementation_contract_tests": "pass",
        "required_command": (
            "RUN_FAZ8_INTEGRATION=1 UI_URL=http://127.0.0.1:7860 "
            "PYTHONPATH=service pytest service/tests/test_t10_ui.py -q"
        ),
        "expected_environment": (
            "healthy canonical Docker Compose stack, active real ingest run with DATA_ROOT local MP4, "
            "API/UI reachable on loopback, Playwright Chromium installed"
        ),
        "image": "artifacts/faz11/ui_smoke.png",
    }
    (ARTIFACT_ROOT / "ui_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
