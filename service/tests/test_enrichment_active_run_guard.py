import pytest

from app.db import clickhouse, postgres


RUN_ID = "00000000-0000-0000-0000-000000000099"
DATASET_ID = "active_enrichment_guard"
ROWS = [("segment-1", 2.0, 0.5)]


@pytest.mark.parametrize("writer,args", [
    (postgres.write_run_detector_enrichment, (RUN_ID, DATASET_ID, ROWS)),
    (clickhouse.write_run_detector_enrichment, (RUN_ID, DATASET_ID, 512, ROWS)),
])
def test_detector_enrichment_cannot_mutate_active_run(monkeypatch, writer, args):
    monkeypatch.setattr(
        postgres, "get_active_run_snapshot",
        lambda dataset_id: {"run_id": RUN_ID} if dataset_id == DATASET_ID else None,
    )
    with pytest.raises(ValueError, match="active run"):
        writer(*args)
