from bench.detector_baseline import GT_CONCEPT_MAP, gt_counts_for_window
from eval.make_groundtruth import CAT


def test_gt_counts_for_window_counts_pedestrian_and_people_as_person():
    frames = {
        1: [(0, CAT["pedestrian"], 0, 0), (1, CAT["people"], 1, 1)],
    }
    counts = gt_counts_for_window(frames, t0=0.0, t1=1.0, fps=25.0, n_sample=1)
    assert counts["person"] == 2.0


def test_gt_counts_for_window_separates_concepts():
    frames = {
        1: [(0, CAT["bus"], 0, 0), (1, CAT["truck"], 1, 1), (2, CAT["car"], 2, 2)],
    }
    counts = gt_counts_for_window(frames, t0=0.0, t1=1.0, fps=25.0, n_sample=1)
    assert counts["bus"] == 1.0
    assert counts["truck"] == 1.0
    assert counts["car"] == 1.0
    assert counts["person"] == 0.0


def test_gt_counts_for_window_empty_frames_is_zero():
    counts = gt_counts_for_window({}, t0=0.0, t1=1.0, fps=25.0, n_sample=3)
    assert all(v == 0.0 for v in counts.values())


def test_gt_concept_map_only_uses_known_columns():
    assert set(GT_CONCEPT_MAP.values()) == {"person", "car", "truck", "bus"}
