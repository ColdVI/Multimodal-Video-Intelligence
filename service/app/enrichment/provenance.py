"""Enrichment provenance stays in its own namespace, never overwriting
embedding_provenance -- plan Sec.8.2: "Embedding provenance alanlari detector
tarafindan ezilmez. Ayni run provenance zinciri altinda ayri namespace kullanilir."
"""

from __future__ import annotations

from typing import Any

from app.enrichment.contracts import DetectorConfig


def build_detector_provenance(
    config: DetectorConfig, *, model_id: str, checkpoint_hash: str, taxonomy: str, class_map_version: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id, "checkpoint_hash": checkpoint_hash, "taxonomy": taxonomy,
        "class_map_version": class_map_version, "confidence": config.confidence,
        "frames_per_window": config.frames_per_window, "aggregation": "median_visible",
        "failure_policy": config.failure_policy,
    }


def merge_provenance(embedding_provenance: dict[str, Any], detector_provenance: dict[str, Any] | None) -> dict[str, Any]:
    result = {"embedding_provenance": embedding_provenance}
    if detector_provenance is not None:
        result["enrichment_provenance"] = {"detector": detector_provenance}
    return result


__all__ = ["build_detector_provenance", "merge_provenance"]
