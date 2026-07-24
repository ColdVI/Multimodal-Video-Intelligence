from eval.make_groundtruth import (CAT, frames_to_intervals, gt_object,
                                    gt_walking, intersect)


def test_frames_to_intervals_merges_short_gap():
    flags = [True] * 25 + [False] * 10 + [True] * 25  # 1sn bosluk, fps=25
    iv = frames_to_intervals(flags, fps=25, min_dur=0.5, gap_tol_s=2.0)
    assert len(iv) == 1


def test_frames_to_intervals_splits_long_gap():
    flags = [True] * 25 + [False] * 100 + [True] * 25  # 4sn bosluk
    iv = frames_to_intervals(flags, fps=25, min_dur=0.5, gap_tol_s=2.0)
    assert len(iv) == 2


def test_frames_to_intervals_drops_short_bursts():
    flags = [False] * 10 + [True] * 5 + [False] * 10  # 0.2sn, min_dur=1.0
    iv = frames_to_intervals(flags, fps=25, min_dur=1.0)
    assert iv == []


def test_gt_object_finds_bus_frames():
    # otobus 30 ardisik karede gorunuyor (1.2sn, 25fps) - tek karelik bir
    # gorunmenin (0.04sn) min_dur=1.0sn esigini gecmemesi DOGRU davranis,
    # bu yuzden gerceklige yakin bir sure kullaniyoruz.
    frames = {i: [(0, CAT["bus"], 100, 100)] for i in range(1, 31)}
    iv = gt_object(frames, n_frames=40, cat_name="bus", fps=25)
    assert len(iv) == 1
    t0, t1 = iv[0]
    assert abs((t1 - t0) - 30 / 25) < 1e-6


def test_gt_walking_detects_displacement():
    frames = {1: [(0, CAT["pedestrian"], 0, 0)],
              26: [(0, CAT["pedestrian"], 0, 100)]}
    iv = gt_walking(frames, n_frames=26, fps=25, px_per_s=15.0)
    assert len(iv) >= 1


def test_gt_walking_ignores_stationary():
    frames = {1: [(0, CAT["pedestrian"], 0, 0)],
              26: [(0, CAT["pedestrian"], 1, 1)]}
    iv = gt_walking(frames, n_frames=26, fps=25, px_per_s=15.0)
    assert iv == []


def test_intersect_overlapping():
    a = [(0.0, 10.0)]
    b = [(5.0, 15.0)]
    out = intersect(a, b, min_overlap=1.0)
    assert out == [(5.0, 10.0)]


def test_intersect_no_overlap_below_min():
    a = [(0.0, 5.0)]
    b = [(4.5, 10.0)]
    out = intersect(a, b, min_overlap=1.0)
    assert out == []
