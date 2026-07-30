"""Pure-function coverage for the redesigned UI layer — no live server needed.

Complements service/tests/test_t10_ui.py (which drives a real browser against a
live stack) with fast checks for the HTML-rendering helpers and the CSV export
path referenced in docs/UI_REGRESSION_REPORT.md.
"""

from __future__ import annotations

import csv

from ui import app as ui_app
from ui import components


def test_score_indicator_renders_dash_for_missing_score():
    assert "—" in components.score_indicator(None)


def test_score_indicator_renders_numeric_value():
    html = components.score_indicator(0.1748)
    assert "0.1748" in html
    assert "score-indicator__fill" in html


def test_media_slot_placeholder_matches_talimat_wording():
    html = components.media_slot({
        "file_path": "/workspace/data/research/auair/images.zip::20190905091750",
        "t_start": 148.0, "t_end": 156.0,
    })
    assert "images.zip::20190905091750" in html
    assert "148.0s" in html and "156.0s" in html
    assert "medya önizlemesi bu ortamda servis edilmiyor" in html
    assert "src=" not in html


def test_media_slot_renders_player_when_src_is_provided():
    html = components.media_slot({"t_start": 0, "t_end": 1}, src="https://example.test/clip.mp4")
    assert "<video" in html
    assert "media-slot--player" in html


def test_diagnostics_panel_uses_olculmedi_for_null_indexed_vectors_not_zero():
    html = components.diagnostics_panel({
        "candidate_count": 1866, "returned_count": 10, "underfilled": False,
        "underfilled_reason": None, "plan_used_vector_index": False,
        "indexed_vectors_count": None, "filter_correctness": True,
        "quality_vs_groundtruth": None, "r_at_1": None, "ndcg": None, "notes": [],
    }, "synthetic")
    assert "ölçülmedi" in html
    assert ">0<" not in html


def test_telemetry_badges_hide_null_gimbal_pitch():
    html = components.telemetry_badges({
        "altitude_m": 10.3, "velocity_mps": 0.06, "gimbal_pitch": None,
    })
    assert "İrtifa" in html
    assert "Gimbal" not in html


def test_result_list_falls_back_to_empty_state_for_no_results():
    html = components.result_list([])
    assert "empty-state" in html
    assert "Sonuç bulunamadı" in html


def test_export_csv_writes_additive_fields_as_columns(tmp_path, monkeypatch):
    raw_response = {
        "results": [{
            "segment_id": "auair:20190905091750:148.000:156.000", "video_id": "20190905091750",
            "t_start": 148.0, "t_end": 156.0, "score": 0.1748, "caption": None,
            "file_path": "/workspace/data/research/auair/images.zip::20190905091750",
            "altitude_m": 10.321, "velocity_mps": 0.0614, "gimbal_pitch": None,
            "event_category": None, "split": "train", "person_count": 1,
            "vehicle_count": 245, "bus_count": 0,
        }],
    }
    monkeypatch.setattr(ui_app.tempfile, "gettempdir", lambda: str(tmp_path))
    path = ui_app.export_csv(raw_response)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["video_id"] == "20190905091750"
    assert set(ui_app.RESULT_COLUMNS) == set(rows[0].keys())
