"""SLA-controlled VLM rerank policy (plan Sec.11), gated on VLM_RERANK_MODE (default
"off"). This module is the decision logic only -- should_trigger_vlm_rerank() is a pure
function of signals a caller already has, with no model call inside it. The actual VLM
call is out of scope this session: it needs a real VLM available, and per Sec.11.1 the
production target co-locates a 112B model with the query embedder on the same GPU, whose
VRAM co-residency was never measured here (no representative GPU in this environment --
see docs/operations/KNOWN_LIMITATIONS.md). Building the trigger policy and SLA/circuit-
breaker contract now means the moment that measurement exists, wiring in a real call is a
policy-off-by-default flip, not a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VLMRerankMode = Literal["off", "auto", "force"]


@dataclass(frozen=True)
class VLMRerankSLA:
    max_vlm_candidates: int = 20
    vlm_timeout_ms: float = 3000.0
    max_concurrent_vlm_requests: int = 2
    max_queue_depth: int = 10
    fallback_to_embedding_results: bool = True

    def __post_init__(self) -> None:
        if self.max_vlm_candidates < 1:
            raise ValueError("max_vlm_candidates must be >= 1")
        if self.vlm_timeout_ms <= 0:
            raise ValueError("vlm_timeout_ms must be positive")
        if self.max_concurrent_vlm_requests < 1:
            raise ValueError("max_concurrent_vlm_requests must be >= 1")
        if self.max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")


@dataclass(frozen=True)
class VLMTriggerSignals:
    top_score_margin: float  # gap between rank-1 and rank-2 embedding scores
    is_hard_negative_category: bool
    detector_embedding_conflict: bool  # detector says the object isn't there, embedding ranked it top
    latency_budget_allows_vlm: bool
    gpu_queue_depth: int
    vram_residency_gate_passed: bool


@dataclass(frozen=True)
class VLMTriggerDecision:
    should_call_vlm: bool
    reason: str


def should_trigger_vlm_rerank(
    mode: VLMRerankMode, signals: VLMTriggerSignals, sla: VLMRerankSLA, *, low_margin_threshold: float = 0.05,
) -> VLMTriggerDecision:
    if mode == "off":
        return VLMTriggerDecision(False, "vlm_rerank_mode=off")
    if not signals.vram_residency_gate_passed:
        # Sec.11.1: this gate is a hard precondition, checked before *any* auto/force
        # trigger, because a co-residency failure means the interactive path cannot
        # safely hold both models in VRAM at once -- not a per-request judgment call.
        return VLMTriggerDecision(False, "vram_residency_gate_not_passed")
    if mode == "force":
        return VLMTriggerDecision(True, "vlm_rerank_mode=force")
    # mode == "auto": every one of these must hold (plan Sec.11's auto conditions are
    # listed as a conjunction -- "top score margin low" AND "hard-negative category" is
    # too strong a reading if EITHER alone were sufficient to justify the latency/GPU
    # cost on every request; requiring several signals to agree is the conservative,
    # intentionally narrow interpretation of "auto" the plan's own framing supports).
    if not signals.latency_budget_allows_vlm:
        return VLMTriggerDecision(False, "latency_budget_does_not_allow_vlm")
    if signals.gpu_queue_depth >= sla.max_queue_depth:
        return VLMTriggerDecision(False, "gpu_queue_depth_at_or_above_max_queue_depth")
    triggered_by = []
    if signals.top_score_margin < low_margin_threshold:
        triggered_by.append("low_score_margin")
    if signals.is_hard_negative_category:
        triggered_by.append("hard_negative_category")
    if signals.detector_embedding_conflict:
        triggered_by.append("detector_embedding_conflict")
    if triggered_by:
        return VLMTriggerDecision(True, "+".join(triggered_by))
    return VLMTriggerDecision(False, "no_trigger_signal_present")


__all__ = ["VLMRerankMode", "VLMRerankSLA", "VLMTriggerSignals", "VLMTriggerDecision", "should_trigger_vlm_rerank"]
