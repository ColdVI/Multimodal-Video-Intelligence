from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import CAPERA_PROTOCOL


DIMENSIONS = (2048, 1024, 512, 256)


def _normalize_prefix(matrix: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(matrix[:, :dimension], dtype=np.float32).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or (norms == 0).any():
        raise ValueError(f"invalid {dimension}d MRL input")
    result /= norms
    return result


def retrieval_ranks(
    item_vectors: np.ndarray,
    query_vectors: np.ndarray,
    relevant_positions: np.ndarray,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Tek relevant-item GT icin stable exact rank; tum skor matrisini saklamaz."""
    ranks = np.empty(len(query_vectors), dtype=np.int32)
    item_positions = np.arange(len(item_vectors))
    for start in range(0, len(query_vectors), block_size):
        stop = min(start + block_size, len(query_vectors))
        scores = query_vectors[start:stop] @ item_vectors.T
        relevant = relevant_positions[start:stop]
        relevant_scores = scores[np.arange(stop - start), relevant]
        greater = (scores > relevant_scores[:, None]).sum(axis=1)
        tied_before = (
            (scores == relevant_scores[:, None])
            & (item_positions[None, :] < relevant[:, None])
        ).sum(axis=1)
        ranks[start:stop] = 1 + greater + tied_before
    return ranks


def per_query_metrics(ranks: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(ranks)
    return {
        "r_at_1": (values <= 1).astype(np.float64),
        "r_at_5": (values <= 5).astype(np.float64),
        "r_at_10": (values <= 10).astype(np.float64),
        "mrr": 1.0 / values,
        "ndcg_at_10": np.where(values <= 10, 1.0 / np.log2(values + 1.0), 0.0),
        "map": 1.0 / values,
    }


def video_cluster_bootstrap_ci(
    differences: np.ndarray,
    video_ids: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Query degil video cluster'larini yeniden ornekleyen paired bootstrap."""
    frame = pd.DataFrame({
        "video_id": np.asarray(video_ids, dtype=str),
        "difference": np.asarray(differences, dtype=np.float64),
    })
    cluster_means = frame.groupby("video_id", sort=True)["difference"].mean().to_numpy()
    if len(cluster_means) < 2:
        raise ValueError("video-level cluster bootstrap requires at least two videos")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    batch_size = 256
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, len(cluster_means), size=(stop - start, len(cluster_means)))
        samples[start:stop] = cluster_means[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "difference": float(cluster_means.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "clusters": int(len(cluster_means)),
        "resamples": int(n_resamples),
    }


def halfvec_quantization_experiment(
    item_vectors: np.ndarray,
    query_vectors: np.ndarray,
    relevant_positions: np.ndarray,
) -> dict[str, Any]:
    """2048d pgvector halfvec etkisini float32 exact esitliginden ayri olcer."""
    float_items = _normalize_prefix(item_vectors, 2048)
    float_queries = _normalize_prefix(query_vectors, 2048)
    half_items = np.asarray(float_items, dtype=np.float16).astype(np.float32)
    half_queries = np.asarray(float_queries, dtype=np.float16).astype(np.float32)
    half_items /= np.linalg.norm(half_items, axis=1, keepdims=True)
    half_queries /= np.linalg.norm(half_queries, axis=1, keepdims=True)
    float_ranks = retrieval_ranks(float_items, float_queries, relevant_positions)
    half_ranks = retrieval_ranks(half_items, half_queries, relevant_positions)
    cosines = np.sum(float_items * half_items, axis=1)
    return {
        "dimension": 2048,
        "storage": "halfvec",
        "mean_item_cosine_after_quantization": float(cosines.mean()),
        "r_at_1_float32": float((float_ranks == 1).mean()),
        "r_at_1_halfvec_simulation": float((half_ranks == 1).mean()),
        "rank_equal_fraction": float((float_ranks == half_ranks).mean()),
        "exact_equality_required": False,
    }


def evaluate_capera(root: Path, *, n_resamples: int = 10_000) -> dict[str, Any]:
    item_vectors = np.load(root / "capera_2048.npy", mmap_mode="r")
    query_vectors = np.load(root / "capera_queries_2048.npy", mmap_mode="r")
    item_ids = pd.read_parquet(root / "capera_ids.parquet")
    query_ids = pd.read_parquet(root / "capera_query_ids.parquet")
    expected_items = int(CAPERA_PROTOCOL["items"])
    expected_queries = int(CAPERA_PROTOCOL["queries"])
    if item_vectors.shape != (expected_items, 2048) or query_vectors.shape != (expected_queries, 2048):
        raise ValueError(f"CapERA quality shape contract failed: {item_vectors.shape}/{query_vectors.shape}")
    segment_to_position = {
        str(segment_id): position
        for position, segment_id in enumerate(item_ids["segment_id"])
    }
    relevant_positions = np.asarray([
        segment_to_position[str(segment_id)]
        for segment_id in query_ids["relevant_segment_id"]
    ], dtype=np.int32)
    videos = query_ids["relevant_video_id"].astype(str).to_numpy()
    by_dimension: dict[str, Any] = {}
    per_dimension: dict[int, dict[str, np.ndarray]] = {}
    for dimension in DIMENSIONS:
        items = _normalize_prefix(item_vectors, dimension)
        queries = _normalize_prefix(query_vectors, dimension)
        ranks = retrieval_ranks(items, queries, relevant_positions)
        per_query = per_query_metrics(ranks)
        per_dimension[dimension] = per_query
        by_dimension[str(dimension)] = {
            metric: float(values.mean()) for metric, values in per_query.items()
        }
    comparisons: dict[str, Any] = {}
    for dimension in (1024, 512, 256):
        comparisons[f"2048_vs_{dimension}"] = {
            metric: video_cluster_bootstrap_ci(
                per_dimension[2048][metric] - per_dimension[dimension][metric],
                videos, n_resamples=n_resamples,
            )
            for metric in per_dimension[2048]
        }
    fixed_path = root / "query_embeddings.json"
    fixed_payload = json.loads(fixed_path.read_text(encoding="utf-8"))
    fixed_queries = fixed_payload.get("queries", fixed_payload)
    exploratory = {
        query_id: {
            "status": "EXPLORATORY",
            "pass_fail": None,
            "reason": "negation/nonsense queries are findings, not quality gates",
            "present_in_fixed_cache": any(
                marker in text.lower() for text in fixed_queries
            ),
        }
        for query_id, marker in (("S17", "no people"), ("S18", "no vehicles"), ("S19", "zzzqqq"))
    }
    return {
        "dataset_id": "capera", "split": CAPERA_PROTOCOL["split"],
        "items": expected_items, "queries": expected_queries,
        "caption_source": "unknown", "by_dimension": by_dimension,
        "paired_video_cluster_bootstrap": comparisons,
        "halfvec_2048_quantization": halfvec_quantization_experiment(
            item_vectors, query_vectors, relevant_positions,
        ),
        "exploratory": exploratory,
    }


__all__ = [
    "evaluate_capera", "halfvec_quantization_experiment", "per_query_metrics",
    "retrieval_ranks", "video_cluster_bootstrap_ci",
]
