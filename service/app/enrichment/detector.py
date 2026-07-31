"""Real detector invocation -- the only module in app/enrichment/ that imports a heavy
model dependency, and only inside functions, never at module load time (mirrors
app/embedding/text_cpu.py's lazy-load pattern exactly).

Config-driven variant resolution mirrors ingest/04_detect.py::_resolve_variant()'s
existing config.yaml: detector.variants registry (plan Sec.8.2: "Model yolu koda
gomulmez"). This module does not import ingest/04_detect.py itself -- the research and
product planes stay import-independent (plan Sec.4/"Araştırma düzlemindeki kodları
production import yoluna baglama") -- it re-reads the same config structure instead of
sharing code across the plane boundary, which is a deliberate, small duplication rather
than a cross-plane import.

NOT executed end-to-end against a real video this session (would need a real detector
run over real frames, which is an ingest-pipeline integration exercise, not a unit-
testable one) -- see docs/operations/KNOWN_LIMITATIONS.md. detector.py's own contract
(config resolution, the DetectionResult shape it must produce for aggregation.py) is
exercised by test_enrichment_detector.py with a fake model.
"""

from __future__ import annotations

from typing import Any, Callable

from app.enrichment.aggregation import DetectionResult
from app.enrichment.contracts import DetectorConfig

_models: dict[str, Any] = {}


def resolve_variant(detector_config: dict[str, Any], variant: str | None = None) -> tuple[str, dict[int, str]]:
    """Same config shape as ingest/04_detect.py::_resolve_variant(): config.yaml's
    detector.variants.<name>.{checkpoint,class_map}. Falls back to the COCO checkpoint/
    class map used there if no variant is configured, for the same reason: existing
    deployments without a detector.variants block must not break."""
    variants = detector_config.get("variants", {})
    variant = variant or detector_config.get("default_variant")
    if variant and variant in variants:
        spec = variants[variant]
        return spec["checkpoint"], {int(k): v for k, v in spec["class_map"].items()}
    return "yolo26x.pt", {0: "person", 2: "car", 5: "bus", 7: "truck"}


def _get_model(checkpoint: str) -> Any:
    if checkpoint not in _models:
        from ultralytics import YOLO

        _models[checkpoint] = YOLO(checkpoint)
    return _models[checkpoint]


def run_detection_on_frames(
    segment_id: str, frames: list[Any], config: DetectorConfig, class_map: dict[int, str],
    checkpoint: str, *, model_loader: Callable[[str], Any] = _get_model,
) -> DetectionResult:
    """model_loader is injected so tests can supply a fake model without ultralytics
    installed -- production callers never pass it, using the real _get_model default."""
    try:
        model = model_loader(checkpoint)
        concepts = sorted(set(class_map.values()))
        counts: dict[str, list[int]] = {concept: [] for concept in concepts}
        confidences: list[float] = []
        for frame in frames:
            predictions = model.predict(frame, conf=config.confidence, verbose=False)[0]
            frame_counts = {concept: 0 for concept in concepts}
            for cls_id, conf in zip(predictions.boxes.cls.tolist(), predictions.boxes.conf.tolist()):
                concept = class_map.get(int(cls_id))
                if concept in frame_counts:
                    frame_counts[concept] += 1
                    confidences.append(float(conf))
            for concept in concepts:
                counts[concept].append(frame_counts[concept])
        return DetectionResult(segment_id=segment_id, class_counts=counts, confidence_distribution=confidences)
    except Exception:
        if config.failure_policy == "strict":
            raise
        return DetectionResult(segment_id=segment_id, class_counts=None, confidence_distribution=[])


__all__ = ["resolve_variant", "run_detection_on_frames"]
