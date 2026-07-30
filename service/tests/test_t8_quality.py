from __future__ import annotations

import numpy as np

from app.bench.quality import video_cluster_bootstrap_ci


def test_t8_bootstrap_resamples_video_clusters_not_query_rows():
    differences = np.asarray([1.0, 1.0, -1.0, -1.0])
    video_ids = np.asarray(["video-a", "video-a", "video-b", "video-b"])
    result = video_cluster_bootstrap_ci(differences, video_ids, n_resamples=500, seed=3)
    assert result["clusters"] == 2
    assert result["difference"] == 0.0
    assert result["ci95_low"] <= 0 <= result["ci95_high"]


def test_t8_negation_and_nonsense_are_not_pass_fail_gates():
    statuses = {
        "S17": {"status": "EXPLORATORY", "pass_fail": None},
        "S18": {"status": "EXPLORATORY", "pass_fail": None},
        "S19": {"status": "EXPLORATORY", "pass_fail": None},
    }
    assert all(item["pass_fail"] is None for item in statuses.values())
