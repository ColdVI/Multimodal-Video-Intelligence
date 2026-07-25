import time

from bench.timing import StageTimer, _percentile


def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([], 50) == 0.0


def test_percentile_interpolates():
    assert _percentile([1, 2], 50) == 1.5


def test_measure_records_duration():
    timer = StageTimer()
    with timer.measure("decode"):
        time.sleep(0.01)
    summary = timer.summary()
    assert summary["decode"]["n"] == 1
    assert summary["decode"]["mean_s"] >= 0.01


def test_measure_accumulates_multiple_calls():
    timer = StageTimer()
    for _ in range(3):
        with timer.measure("query"):
            pass
    assert timer.summary()["query"]["n"] == 3


def test_measure_keeps_stages_separate():
    timer = StageTimer()
    with timer.measure("decode"):
        pass
    with timer.measure("embed"):
        pass
    assert set(timer.summary()) == {"decode", "embed"}
