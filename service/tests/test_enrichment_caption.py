from __future__ import annotations

import pytest

from app.enrichment.caption import assert_caption_not_authoritative, run_caption


def test_forbidden_fields_raise():
    for field in ("velocity_mps", "altitude_m", "person_count", "vehicle_count", "bus_count", "is_night"):
        with pytest.raises(ValueError, match="never authoritatively"):
            assert_caption_not_authoritative(field)


def test_non_telemetry_field_is_not_blocked():
    assert_caption_not_authoritative("event_category")  # does not raise


def test_run_caption_off_mode_is_rejected():
    with pytest.raises(ValueError, match="caption_mode='off'"):
        run_caption("s1", "off", lambda seg: "text")


def test_run_caption_success():
    record = run_caption("s1", "sampled", lambda seg: f"a drone view of {seg}")
    assert record.text == "a drone view of s1"


def test_run_caption_model_failure_does_not_propagate():
    def failing(seg):
        raise RuntimeError("model unavailable")

    record = run_caption("s1", "sampled", failing)
    assert record.text is None
