"""Asama bazli duvar-saati (+ varsa CUDA event) zamanlayici (Faz 1 madde 1):
decode, YOLO, embed, CH load, query. time.perf_counter taban; GPU'da
torch.cuda.synchronize sonrasi olcer. torch import edilemezse (bozuk/eksik
ortam) CUDA senkronizasyonu sessizce atlanir, duvar-saati olcumu etkilenmez."""
import time
from contextlib import contextmanager

try:
    import torch
except Exception:  # pragma: no cover - bench her ortamda torch bulamayabilir
    torch = None


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(durations: list) -> dict:
    mean = sum(durations) / len(durations)
    variance = sum((d - mean) ** 2 for d in durations) / len(durations)
    return {
        "n": len(durations),
        "total_s": sum(durations),
        "mean_s": mean,
        "std_s": variance ** 0.5,
        "p50_s": _percentile(durations, 50),
        "p95_s": _percentile(durations, 95),
    }


class StageTimer:
    """decode/YOLO/embed/CH-load/query gibi asamalari kaydeder; ayni asama
    birden fazla kez olculebilir (ör. sorgu basina bir kez)."""

    def __init__(self):
        self.stages: dict = {}

    @contextmanager
    def measure(self, stage: str):
        if _cuda_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if _cuda_available():
                torch.cuda.synchronize()
            self.stages.setdefault(stage, []).append(time.perf_counter() - t0)

    def summary(self) -> dict:
        return {stage: _stats(durations) for stage, durations in self.stages.items()}
