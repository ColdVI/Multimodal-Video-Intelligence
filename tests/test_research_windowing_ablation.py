from __future__ import annotations

import pytest

from bench.windowing_ablation import (
    AblationVerdict, WindowConfig, build_sweep, judge_priority_gate,
    run_screening, select_confirmation_candidates,
)


def test_build_sweep_is_the_cartesian_product_and_deduplicated():
    configs = build_sweep({
        "window_size_s": (4.0, 8.0), "stride_s": (2.0, 4.0),
        "n_sample": (6,), "gap_tolerance_s": (10.0,), "iou_threshold": (0.5,),
    })
    assert len(configs) == 4
    assert len({c.config_id for c in configs}) == 4


def test_config_id_is_stable_and_human_readable():
    config = WindowConfig(window_size_s=8.0, stride_s=4.0, n_sample=6, gap_tolerance_s=10.0, iou_threshold=0.5)
    assert config.config_id == "w8_s4_n6_g10_iou0.5"


def test_run_screening_sorts_best_first_by_primary_metric():
    configs = build_sweep({
        "window_size_s": (4.0, 8.0), "stride_s": (2.0,), "n_sample": (6,),
        "gap_tolerance_s": (10.0,), "iou_threshold": (0.5,),
    })
    scores = {configs[0].config_id: 0.30, configs[1].config_id: 0.55}

    results = run_screening(configs, lambda c: {"recall_at_10": scores[c.config_id]})
    assert [r.config.config_id for r in results] == sorted(scores, key=scores.get, reverse=True)


def test_select_confirmation_candidates_takes_top_n_after_sorting():
    configs = build_sweep({
        "window_size_s": (2.0, 4.0, 8.0), "stride_s": (1.0,), "n_sample": (6,),
        "gap_tolerance_s": (10.0,), "iou_threshold": (0.5,),
    })
    scores = {configs[0].config_id: 0.1, configs[1].config_id: 0.9, configs[2].config_id: 0.5}
    results = run_screening(configs, lambda c: {"recall_at_10": scores[c.config_id]})
    top2 = select_confirmation_candidates(results, top_n=2)
    assert [c.config_id for c in top2] == [configs[1].config_id, configs[2].config_id]


def test_select_confirmation_candidates_rejects_non_positive_top_n():
    with pytest.raises(ValueError):
        select_confirmation_candidates([], top_n=0)


def test_judge_priority_gate_flags_large_spread_as_dominant():
    configs = build_sweep({
        "window_size_s": (4.0, 8.0), "stride_s": (2.0,), "n_sample": (6,),
        "gap_tolerance_s": (10.0,), "iou_threshold": (0.5,),
    })
    scores = {configs[0].config_id: 0.30, configs[1].config_id: 0.55}
    results = run_screening(configs, lambda c: {"recall_at_10": scores[c.config_id]})
    verdict = judge_priority_gate(results, threshold=0.05)
    assert isinstance(verdict, AblationVerdict)
    assert verdict.windowing_is_dominant_axis is True
    assert verdict.variance_across_screening == pytest.approx(0.25, abs=1e-9)


def test_judge_priority_gate_does_not_flag_small_spread():
    configs = build_sweep({
        "window_size_s": (4.0, 8.0), "stride_s": (2.0,), "n_sample": (6,),
        "gap_tolerance_s": (10.0,), "iou_threshold": (0.5,),
    })
    scores = {configs[0].config_id: 0.50, configs[1].config_id: 0.51}
    results = run_screening(configs, lambda c: {"recall_at_10": scores[c.config_id]})
    verdict = judge_priority_gate(results, threshold=0.05)
    assert verdict.windowing_is_dominant_axis is False


def test_judge_priority_gate_rejects_empty_results():
    with pytest.raises(ValueError):
        judge_priority_gate([])
