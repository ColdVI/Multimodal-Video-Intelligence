from __future__ import annotations

import pytest

from app.enrichment.contracts import DetectorConfig
from app.enrichment.detector import resolve_variant, run_detection_on_frames

CLASS_MAP = {0: "person", 1: "car"}


def test_resolve_variant_uses_configured_checkpoint_and_class_map():
    config = {"default_variant": "v1", "variants": {"v1": {"checkpoint": "custom.pt", "class_map": {"0": "car"}}}}
    checkpoint, class_map = resolve_variant(config, "v1")
    assert checkpoint == "custom.pt"
    assert class_map == {0: "car"}


def test_resolve_variant_falls_back_to_coco_when_unconfigured():
    checkpoint, class_map = resolve_variant({})
    assert checkpoint == "yolo26x.pt"
    assert class_map[2] == "car"


class _FakeBoxes:
    def __init__(self, cls_ids, confs):
        self._cls_ids = cls_ids
        self._confs = confs

    class _Tensor(list):
        def tolist(self):
            return list(self)

    @property
    def cls(self):
        return self._Tensor(self._cls_ids)

    @property
    def conf(self):
        return self._Tensor(self._confs)


class _FakePrediction:
    def __init__(self, cls_ids, confs):
        self.boxes = _FakeBoxes(cls_ids, confs)


class _FakeModel:
    def __init__(self, per_frame_detections):
        self._per_frame_detections = per_frame_detections
        self.calls = 0

    def predict(self, frame, conf, verbose):
        detections = self._per_frame_detections[self.calls]
        self.calls += 1
        return [_FakePrediction(*detections)]


def test_run_detection_on_frames_counts_per_class_per_frame():
    model = _FakeModel([
        ([0, 1], [0.9, 0.8]),  # frame 0: 1 person, 1 car
        ([1, 1], [0.7, 0.6]),  # frame 1: 2 cars
    ])
    config = DetectorConfig(confidence=0.25, failure_policy="best_effort")
    result = run_detection_on_frames(
        "seg1", [object(), object()], config, CLASS_MAP, "fake.pt", model_loader=lambda checkpoint: model,
    )
    assert result.class_counts == {"person": [1, 0], "car": [1, 2]}
    assert result.confidence_distribution == [0.9, 0.8, 0.7, 0.6]


def test_run_detection_best_effort_swallows_model_error_and_returns_none_counts():
    def raising_loader(checkpoint):
        raise RuntimeError("model file missing")

    config = DetectorConfig(failure_policy="best_effort")
    result = run_detection_on_frames("seg1", [object()], config, CLASS_MAP, "missing.pt", model_loader=raising_loader)
    assert result.class_counts is None


def test_run_detection_strict_policy_propagates_the_error():
    def raising_loader(checkpoint):
        raise RuntimeError("model file missing")

    config = DetectorConfig(failure_policy="strict")
    with pytest.raises(RuntimeError, match="model file missing"):
        run_detection_on_frames("seg1", [object()], config, CLASS_MAP, "missing.pt", model_loader=raising_loader)
