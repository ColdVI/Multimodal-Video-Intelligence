"""Two-level windowing ablation harness, per
docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md Sec.5/9.1.

Screening sweeps window/stride/n_sample/gap_tolerance/IoU-threshold configurations with a
fast, already-working embedding path (X-CLIP or SigLIP2 -- both run practically on CPU per
docs/operations/STATUS.md); Confirmation re-checks only the best 2-3 screening candidates
with Qwen3-VL-Embedding. Changing window/stride changes segment boundaries, which means new
embeddings for every sequence under test -- there is no way to compute this from data
already on disk, so a full sweep is a real, budgeted re-ingest cost, not a free query over
existing artifacts.

This module provides the sweep-grid contract and the screening/confirmation gate logic
(pure functions, fully testable without running any real ingest). Actually executing a
config still means calling the existing ingest pipeline (ingest/01_frames_to_video.py,
02_windowing.py, 03_embed.py) with an overridden config.yaml `window`/`merge`/`n_sample`
block per candidate -- run_screening_config()/run_confirmation_config() below are the
integration points for that, deliberately left as an injected `run_ingest_and_eval`
callable so tests never need a real video/model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable


@dataclass(frozen=True)
class WindowConfig:
    window_size_s: float
    stride_s: float
    n_sample: int
    gap_tolerance_s: float
    iou_threshold: float

    @property
    def config_id(self) -> str:
        return (
            f"w{self.window_size_s:g}_s{self.stride_s:g}_n{self.n_sample}"
            f"_g{self.gap_tolerance_s:g}_iou{self.iou_threshold:g}"
        )


DEFAULT_SCREENING_GRID = {
    "window_size_s": (4.0, 8.0),
    "stride_s": (2.0, 4.0),
    "n_sample": (6,),
    "gap_tolerance_s": (10.0,),
    "iou_threshold": (0.5,),
}


def build_sweep(grid: dict[str, tuple] = DEFAULT_SCREENING_GRID) -> list[WindowConfig]:
    """Cartesian product of the grid, deduplicated and deterministically ordered. Mirrors
    the plan's example axis values (4s/2s, 4s/4s, 8s/4s, 8s/8s, 2s/1s) as a *default*, not
    a hardcoded requirement -- callers pick their own grid for a real sweep."""
    keys = list(grid)
    configs = [
        WindowConfig(**dict(zip(keys, combination)))
        for combination in product(*(grid[key] for key in keys))
    ]
    seen: dict[str, WindowConfig] = {}
    for config in configs:
        seen.setdefault(config.config_id, config)
    return list(seen.values())


@dataclass(frozen=True)
class ScreeningResult:
    config: WindowConfig
    metrics: dict[str, float]  # e.g. recall_at_10, mrr, tr_en_agreement -- caller-defined


def run_screening(
    configs: list[WindowConfig],
    run_and_eval: Callable[[WindowConfig], dict[str, float]],
    *,
    primary_metric: str = "recall_at_10",
) -> list[ScreeningResult]:
    """run_and_eval is the injected integration point: in production it re-ingests a
    representative sequence subset at `config` and evaluates against GT, returning a
    metrics dict. Returns results sorted best-first by primary_metric."""
    results = [ScreeningResult(config, run_and_eval(config)) for config in configs]
    return sorted(results, key=lambda result: result.metrics.get(primary_metric, float("-inf")), reverse=True)


def select_confirmation_candidates(
    screening_results: list[ScreeningResult], *, top_n: int = 3,
) -> list[WindowConfig]:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    return [result.config for result in screening_results[:top_n]]


@dataclass(frozen=True)
class AblationVerdict:
    windowing_is_dominant_axis: bool
    variance_across_screening: float
    reason: str


def judge_priority_gate(
    screening_results: list[ScreeningResult], *, primary_metric: str = "recall_at_10", threshold: float = 0.05,
) -> AblationVerdict:
    """Plan Sec.5 kapisi: if windowing produces a bigger recall swing across the
    screening grid than MRL/parser choices are expected to (threshold is the caller's
    own judgment call about what 'bigger' means for their metric -- 0.05 recall points
    is this module's default, not a claim about any other axis's actual measured
    variance), Sec.14's priority order should be revisited before investing further in
    parser/relaxation/detector work. This function only computes the observed spread; it
    never claims a comparison against unmeasured axes.
    """
    if not screening_results:
        raise ValueError("screening_results must be non-empty")
    values = [result.metrics.get(primary_metric, 0.0) for result in screening_results]
    spread = max(values) - min(values)
    dominant = spread >= threshold
    reason = (
        f"{primary_metric} spread across {len(screening_results)} screened configs is "
        f"{spread:.4f} ({'>=' if dominant else '<'} threshold {threshold:.4f})"
    )
    return AblationVerdict(windowing_is_dominant_axis=dominant, variance_across_screening=spread, reason=reason)


__all__ = [
    "WindowConfig", "DEFAULT_SCREENING_GRID", "build_sweep",
    "ScreeningResult", "run_screening", "select_confirmation_candidates",
    "AblationVerdict", "judge_priority_gate",
]
