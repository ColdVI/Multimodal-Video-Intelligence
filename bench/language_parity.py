"""TR/EN cross-lingual text-embedding alignment, per
docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md Sec.6.5/9.2.

Measures whether the Qwen3-VL-Embedding text tower places a Turkish query and its
English translation close together in embedding space, using the existing matched-pair
fixture at tests/fixtures/queries_semantic.json. This does not require a video corpus or
groundtruth: cross-lingual embedding alignment (same-pair cosine similarity + nearest-
neighbor retrieval accuracy across languages) is a standard, much cheaper proxy for "does
the text tower understand Turkish about as well as English" than a full TR-query vs
EN-query retrieval comparison would be, and is measurable on CPU in minutes rather than
requiring the GPU-scale video embedding a full retrieval run would need.

Deliberately does not decide whether structured-filter parsing should happen before or
after translation (plan Sec.6.5) -- it only measures the one input every later decision
depends on: is the semantic residual itself equally usable in both languages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "queries_semantic.json"


@dataclass(frozen=True)
class QueryPair:
    query_id: str
    tr: str
    en: str
    query_type: str
    evaluation: str  # "binding" | "exploratory"


def load_pairs(path: Path = DEFAULT_FIXTURE_PATH) -> list[QueryPair]:
    """Only entries with non-empty tr AND en text are usable for cross-lingual alignment
    (S20 is an intentional empty-string edge case, S23 has no tr/en fields at all -- both
    are skipped here, not silently zero-filled)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for entry in raw:
        tr, en = entry.get("tr"), entry.get("en")
        if not tr or not en:
            continue
        pairs.append(QueryPair(
            query_id=entry["id"], tr=tr, en=en, query_type=entry.get("type", "unknown"),
            evaluation=entry.get("evaluation", "binding"),
        ))
    return pairs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def measure_alignment(
    pairs: Iterable[QueryPair], embed_fn: Callable[[str], np.ndarray],
) -> dict:
    """embed_fn is injected so this is testable without a real Qwen model (see
    tests/test_research_language_parity.py) and reusable for any future text encoder
    candidate from Sec.6.5's ablation (direct TR, TR->EN translation, multilingual
    rewrite) -- swap embed_fn, keep the metric."""
    pairs = list(pairs)
    tr_vectors = [embed_fn(pair.tr) for pair in pairs]
    en_vectors = [embed_fn(pair.en) for pair in pairs]

    per_pair = []
    tr_to_en_correct = 0
    en_to_tr_correct = 0
    for i, pair in enumerate(pairs):
        same_pair_cosine = _cosine(tr_vectors[i], en_vectors[i])
        tr_similarities = [_cosine(tr_vectors[i], en_vectors[j]) for j in range(len(pairs))]
        en_similarities = [_cosine(en_vectors[i], tr_vectors[j]) for j in range(len(pairs))]
        tr_nearest = int(np.argmax(tr_similarities))
        en_nearest = int(np.argmax(en_similarities))
        if tr_nearest == i:
            tr_to_en_correct += 1
        if en_nearest == i:
            en_to_tr_correct += 1
        per_pair.append({
            "query_id": pair.query_id, "query_type": pair.query_type, "evaluation": pair.evaluation,
            "same_pair_cosine": round(same_pair_cosine, 4),
            "tr_nearest_en_query_id": pairs[tr_nearest].query_id, "tr_to_en_top1_correct": tr_nearest == i,
            "en_nearest_tr_query_id": pairs[en_nearest].query_id, "en_to_tr_top1_correct": en_nearest == i,
        })

    def _summary(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0, "mean_same_pair_cosine": None, "tr_to_en_top1_accuracy": None, "en_to_tr_top1_accuracy": None}
        n = len(rows)
        return {
            "n": n,
            "mean_same_pair_cosine": round(float(np.mean([row["same_pair_cosine"] for row in rows])), 4),
            "tr_to_en_top1_accuracy": round(sum(row["tr_to_en_top1_correct"] for row in rows) / n, 4),
            "en_to_tr_top1_accuracy": round(sum(row["en_to_tr_top1_correct"] for row in rows) / n, 4),
        }

    binding_rows = [row for row in per_pair if row["evaluation"] == "binding"]
    exploratory_rows = [row for row in per_pair if row["evaluation"] != "binding"]
    return {
        "per_pair": per_pair,
        "binding": _summary(binding_rows),
        "exploratory": _summary(exploratory_rows),
        "overall": _summary(per_pair),
    }


__all__ = ["QueryPair", "load_pairs", "measure_alignment", "DEFAULT_FIXTURE_PATH"]
