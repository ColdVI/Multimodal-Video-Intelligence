from app.bench.runner import MANDATORY_COLUMNS, configurations, empty_row


def test_l2_matrix_has_at_least_150_rows():
    assert len(list(configurations()))==150


def test_synthetic_quality_is_null_and_mode_present(monkeypatch):
    row=empty_row("clickhouse","exact",512,1.0,"auair",200)
    assert set(MANDATORY_COLUMNS)==set(row)
    assert row["embedding_mode"] and row["quality_vs_groundtruth"] is None
