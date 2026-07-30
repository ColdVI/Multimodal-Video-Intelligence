from __future__ import annotations

from typing import Any


FLOAT32_EXACT_DIMENSIONS = (1024, 512, 256)
HALFVEC_DIMENSIONS = (2048,)


def result_interpretable(embedding_mode: str, strategy: str, metric: str) -> bool:
    if embedding_mode == "synthetic":
        if metric == "ann_recall_vs_exact" and strategy != "exact":
            return False
        if metric in {"quality_vs_groundtruth", "r_at_1", "ndcg", "map"}:
            return False
    return True


def underfilled_classification(candidate_count: int, returned_count: int, top_k: int) -> str | None:
    if returned_count >= top_k:
        return None
    return "candidate_shortage" if candidate_count < top_k else "ann_filter_loss"


def pattern_skip_reason(pattern_execution_implemented: bool) -> str | None:
    return None if pattern_execution_implemented else "pattern not implemented"


def baseline_comparable(current: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        current.get("embedding_mode") == baseline.get("embedding_mode")
        and current.get("hardware_profile") == baseline.get("hardware_profile")
    )


__all__ = [
    "FLOAT32_EXACT_DIMENSIONS", "HALFVEC_DIMENSIONS", "baseline_comparable",
    "pattern_skip_reason", "result_interpretable", "underfilled_classification",
]
