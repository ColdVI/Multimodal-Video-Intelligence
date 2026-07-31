from __future__ import annotations

import pytest

from app.search.vlm_rerank import VLMRerankSLA, VLMTriggerSignals, should_trigger_vlm_rerank


def _signals(**overrides) -> VLMTriggerSignals:
    base = dict(
        top_score_margin=0.5, is_hard_negative_category=False, detector_embedding_conflict=False,
        latency_budget_allows_vlm=True, gpu_queue_depth=0, vram_residency_gate_passed=True,
    )
    base.update(overrides)
    return VLMTriggerSignals(**base)


def test_mode_off_never_triggers_regardless_of_signals():
    decision = should_trigger_vlm_rerank("off", _signals(top_score_margin=0.0, is_hard_negative_category=True), VLMRerankSLA())
    assert decision.should_call_vlm is False
    assert decision.reason == "vlm_rerank_mode=off"


def test_vram_gate_blocks_even_force_mode():
    decision = should_trigger_vlm_rerank("force", _signals(vram_residency_gate_passed=False), VLMRerankSLA())
    assert decision.should_call_vlm is False
    assert decision.reason == "vram_residency_gate_not_passed"


def test_force_mode_always_triggers_when_vram_gate_passes():
    decision = should_trigger_vlm_rerank("force", _signals(), VLMRerankSLA())
    assert decision.should_call_vlm is True
    assert decision.reason == "vlm_rerank_mode=force"


def test_auto_mode_does_not_trigger_with_no_signal_present():
    decision = should_trigger_vlm_rerank("auto", _signals(), VLMRerankSLA())
    assert decision.should_call_vlm is False
    assert decision.reason == "no_trigger_signal_present"


def test_auto_mode_triggers_on_low_score_margin():
    decision = should_trigger_vlm_rerank("auto", _signals(top_score_margin=0.01), VLMRerankSLA())
    assert decision.should_call_vlm is True
    assert "low_score_margin" in decision.reason


def test_auto_mode_triggers_on_hard_negative_category():
    decision = should_trigger_vlm_rerank("auto", _signals(is_hard_negative_category=True), VLMRerankSLA())
    assert decision.should_call_vlm is True
    assert "hard_negative_category" in decision.reason


def test_auto_mode_respects_latency_budget_regardless_of_other_signals():
    decision = should_trigger_vlm_rerank(
        "auto", _signals(top_score_margin=0.0, latency_budget_allows_vlm=False), VLMRerankSLA(),
    )
    assert decision.should_call_vlm is False
    assert decision.reason == "latency_budget_does_not_allow_vlm"


def test_auto_mode_respects_gpu_queue_depth():
    decision = should_trigger_vlm_rerank(
        "auto", _signals(top_score_margin=0.0, gpu_queue_depth=10), VLMRerankSLA(max_queue_depth=10),
    )
    assert decision.should_call_vlm is False
    assert decision.reason == "gpu_queue_depth_at_or_above_max_queue_depth"


def test_multiple_trigger_signals_are_all_reported():
    decision = should_trigger_vlm_rerank(
        "auto", _signals(top_score_margin=0.0, is_hard_negative_category=True, detector_embedding_conflict=True), VLMRerankSLA(),
    )
    assert decision.should_call_vlm is True
    assert decision.reason == "low_score_margin+hard_negative_category+detector_embedding_conflict"


def test_sla_rejects_invalid_values():
    with pytest.raises(ValueError):
        VLMRerankSLA(max_vlm_candidates=0)
    with pytest.raises(ValueError):
        VLMRerankSLA(vlm_timeout_ms=0)
    with pytest.raises(ValueError):
        VLMRerankSLA(max_concurrent_vlm_requests=0)
    with pytest.raises(ValueError):
        VLMRerankSLA(max_queue_depth=-1)
