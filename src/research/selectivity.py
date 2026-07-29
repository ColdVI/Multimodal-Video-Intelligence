"""Histogramdan filtre esigi turetme (spec SS4.2 adim 8, SS5.2). AU-AIR
irtifasi 5-30m dar aralikta oldugu icin (spec SS0 bulgu #2) sabit esik
YAZILAMAZ - her seviye (%50/%10/%1/%0.1) gercek dagilimdan turetilir."""
import numpy as np

DEFAULT_LEVELS = (0.50, 0.10, 0.01, 0.001)


def derive_thresholds(values: list, levels: tuple = DEFAULT_LEVELS, direction: str = "less_than") -> dict:
    """direction='less_than': esigin ALTINDA kalan oran ~= p (ör. altitude_m < theta).
    direction='greater_than': esigin USTUNDE kalan oran ~= p (ör. velocity_mps > theta).
    Donus: {p: {"threshold": float, "actual_selectivity": float, "n": int}}."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    n = len(arr)
    out = {}
    for p in levels:
        if n == 0:
            out[p] = {"threshold": None, "actual_selectivity": None, "n": 0}
            continue
        if direction == "less_than":
            theta = float(np.quantile(arr, p))
            actual = float(np.mean(arr < theta))
        elif direction == "greater_than":
            theta = float(np.quantile(arr, 1 - p))
            actual = float(np.mean(arr > theta))
        else:
            raise ValueError(f"bilinmeyen direction: {direction}")
        out[p] = {"threshold": theta, "actual_selectivity": actual, "n": n}
    return out


__all__ = ["derive_thresholds", "DEFAULT_LEVELS"]
