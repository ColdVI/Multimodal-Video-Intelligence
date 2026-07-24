from search.merge import fmt, merge_intervals


def test_adjacent_windows_merge_within_gap():
    rows = [("v1", 0.0, 8.0, 0.1), ("v1", 4.0, 12.0, 0.2),
            ("v1", 8.0, 16.0, 0.05)]
    out = merge_intervals(rows, gap_tol=10.0)
    assert len(out) == 1
    vid, t0, t1, score = out[0]
    assert vid == "v1" and t0 == 0.0 and t1 == 16.0
    assert score == max(1 - 0.1, 1 - 0.2, 1 - 0.05)


def test_far_windows_stay_separate():
    rows = [("v1", 0.0, 8.0, 0.1), ("v1", 100.0, 108.0, 0.1)]
    out = merge_intervals(rows, gap_tol=10.0)
    assert len(out) == 2


def test_different_videos_never_merge():
    rows = [("v1", 0.0, 8.0, 0.1), ("v2", 0.0, 8.0, 0.1)]
    out = merge_intervals(rows, gap_tol=10.0)
    assert {r[0] for r in out} == {"v1", "v2"}


def test_min_score_filters():
    rows = [("v1", 0.0, 8.0, 0.9), ("v1", 4.0, 12.0, 0.1)]
    out = merge_intervals(rows, gap_tol=10.0, min_score=0.5)
    assert len(out) == 1
    assert out[0][3] == 1 - 0.1


def test_results_sorted_by_score_desc():
    rows = [("v1", 0.0, 8.0, 0.5), ("v2", 0.0, 8.0, 0.1)]
    out = merge_intervals(rows, gap_tol=10.0)
    assert out[0][0] == "v2"  # dist 0.1 -> score 0.9, en yuksek


def test_fmt_hms():
    assert fmt(0) == "0:00:00"
    assert fmt(75) == "0:01:15"
    assert fmt(3725) == "1:02:05"
