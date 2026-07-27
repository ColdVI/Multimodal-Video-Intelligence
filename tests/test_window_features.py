"""ingest/04_detect.py::window_features() icin - gercek YOLO agirligi
yuklemeden, sahte bir model nesnesiyle. Odak: frames= paylasilan-kare
yolu ve BGR/RGB kanal siparisi (bkz. ingest/frame_io.py - RGB donduruyor,
ultralytics ham numpy array'i HER ZAMAN BGR sayip kendi icinde ters
ceviriyor - model'e verilmeden once BGR'ye geri cevrilmis olmali)."""
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "ingest" / "04_detect.py"
SPEC = importlib.util.spec_from_file_location("detect04", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeBoxes:
    def __init__(self, cls_ids):
        self._cls_ids = cls_ids

    @property
    def cls(self):
        class _T:
            def __init__(self, arr):
                self._arr = arr

            def cpu(self):
                return self

            def numpy(self):
                return np.array(self._arr)
        return _T(self._cls_ids)


class _FakeResult:
    def __init__(self, cls_ids):
        self.boxes = _FakeBoxes(cls_ids)


class _FakeModel:
    """cagrildiginda aldigi frame'i kaydeder (kanal sirasi kontrolu icin)
    ve sabit bir tespit dondurur."""

    def __init__(self):
        self.received_frames = []

    def __call__(self, frame, verbose=False):
        self.received_frames.append(frame)
        return [_FakeResult([0])]  # her karede 1 person


def test_window_features_uses_frames_param_without_opening_video(monkeypatch):
    fake_model = _FakeModel()
    MODULE._models["fake.pt"] = fake_model

    # Kirmizi kanali frame indeksine gore ayarlanmis RGB kareler - frame_io
    # ile ayni sozlesme (RGB, kronolojik sirali).
    frames = [np.full((4, 4, 3), 0, dtype=np.uint8),
              np.full((4, 4, 3), 100, dtype=np.uint8),
              np.full((4, 4, 3), 200, dtype=np.uint8)]
    for i, f in enumerate(frames):
        f[:, :, 0] = i * 50  # R kanali

    result = MODULE.window_features(
        "unused-path-should-not-be-opened.mp4", t0=0.0, t1=1.0, n_sample=3,
        checkpoint="fake.pt", class_map={0: "person"}, frames=frames)

    assert result["person_count"] == 1
    # Model'e verilen frame BGR olmali: frame_io RGB donduruyor, ultralytics
    # ham array'i HER ZAMAN BGR sanip kendi icinde ters ceviriyor - RGB'yi
    # oldugu gibi versek kanallar bozulurdu.
    for original, received in zip(frames, fake_model.received_frames):
        assert np.array_equal(received, original[..., ::-1])

    del MODULE._models["fake.pt"]


def test_window_features_frames_param_subsamples_when_more_than_n_sample(monkeypatch):
    fake_model = _FakeModel()
    MODULE._models["fake2.pt"] = fake_model

    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(10)]
    MODULE.window_features(
        "unused.mp4", t0=0.0, t1=1.0, n_sample=3,
        checkpoint="fake2.pt", class_map={0: "person"}, frames=frames)

    assert len(fake_model.received_frames) == 3

    del MODULE._models["fake2.pt"]
