import pytest

from reports.scope_badge import render_scope_badge


def test_render_includes_kind_dataset_count_and_purpose():
    out = render_scope_badge(
        kind="SMOKE_TEST", dataset="VisDrone (5 sekans smoke)", count=14,
        count_label="satır (2 tablo x 7)", purpose="ClickHouse sorgu şekli sağlaması",
        can_claim=["SQL sözdizimi çalışıyor"], cannot_claim=["gerçek ölçekte gecikme"])
    assert "SMOKE TEST" in out
    assert "VisDrone (5 sekans smoke)" in out
    assert "14" in out
    assert "ClickHouse sorgu şekli sağlaması" in out
    assert "SQL sözdizimi çalışıyor" in out
    assert "gerçek ölçekte gecikme" in out


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        render_scope_badge(kind="NOPE", dataset="x", count=1, count_label="x",
                          purpose="x", can_claim=[], cannot_claim=[])


def test_all_four_kinds_render_without_error():
    for kind in ("SMOKE_TEST", "REAL", "SYNTHETIC", "PILOT"):
        out = render_scope_badge(kind=kind, dataset="x", count=1, count_label="x",
                                 purpose="x", can_claim=["a"], cannot_claim=["b"])
        assert kind.split("_")[0] in out or "GERÇEK" in out or "SENTETİK" in out or "PİLOT" in out


def test_html_escapes_dataset_and_claims():
    out = render_scope_badge(kind="REAL", dataset="<script>alert(1)</script>", count=1,
                             count_label="x", purpose="x",
                             can_claim=["<b>bold</b>"], cannot_claim=[])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_generated_at_included_when_provided():
    out = render_scope_badge(kind="REAL", dataset="x", count=1, count_label="x",
                             purpose="x", can_claim=[], cannot_claim=[],
                             generated_at="2026-07-24")
    assert "2026-07-24" in out


def test_generated_at_omitted_when_not_provided():
    out = render_scope_badge(kind="REAL", dataset="x", count=1, count_label="x",
                             purpose="x", can_claim=[], cannot_claim=[])
    assert out.count("<span") == 1  # sadece baslik etiketi, tarih span'i yok


def test_count_is_thousands_separated():
    out = render_scope_badge(kind="SYNTHETIC", dataset="x", count=100000,
                             count_label="satır", purpose="x", can_claim=[], cannot_claim=[])
    assert "100,000" in out
