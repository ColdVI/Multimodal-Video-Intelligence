from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.readiness_check import validate_capera_artifacts


def _unit_vectors(rows: int) -> np.ndarray:
    values = np.zeros((rows, 2048), dtype=np.float32)
    values[:, 0] = 1.0
    return values


def test_capera_artifact_contract_accepts_split_qualified_unknown_provenance(tmp_path):
    np.save(tmp_path / "capera_2048.npy", _unit_vectors(2))
    np.save(tmp_path / "capera_queries_2048.npy", _unit_vectors(4))
    pd.DataFrame({"segment_id": [
        "capera:test__a.mp4:0.000:5.000", "capera:test__b.mp4:0.000:5.000",
    ]}).to_parquet(tmp_path / "capera_ids.parquet", index=False)
    pd.DataFrame({
        "query_id": [f"q{i}" for i in range(4)],
        "query_text": [f"text{i}" for i in range(4)],
        "relevant_segment_id": [
            "capera:test__a.mp4:0.000:5.000", "capera:test__a.mp4:0.000:5.000",
            "capera:test__b.mp4:0.000:5.000", "capera:test__b.mp4:0.000:5.000",
        ],
        "relevant_video_id": ["test__a.mp4", "test__a.mp4", "test__b.mp4", "test__b.mp4"],
        "caption_index": [0, 1, 0, 1],
        "caption_source": ["unknown"] * 4,
    }).to_parquet(tmp_path / "capera_query_ids.parquet", index=False)
    (tmp_path / "embedding_manifest.json").write_text(json.dumps({
        "item_count": 2, "query_count": 4, "split": "test",
        "embedding_mode": "real", "model_revision": "revision",
    }), encoding="utf-8")
    (tmp_path / "query_embeddings.json").write_text(json.dumps({
        "model_revision": "revision", "queries": {"demo": _unit_vectors(1)[0].tolist()},
    }), encoding="utf-8")

    result = validate_capera_artifacts(tmp_path, {
        "split": "test", "items": 2, "captions_per_item": 2, "queries": 4,
    })
    assert result["ok"] is True


def test_capera_artifact_contract_rejects_invented_caption_provenance(tmp_path):
    np.save(tmp_path / "capera_2048.npy", _unit_vectors(1))
    np.save(tmp_path / "capera_queries_2048.npy", _unit_vectors(1))
    pd.DataFrame({"segment_id": ["capera:test__a.mp4:0.000:5.000"]}).to_parquet(
        tmp_path / "capera_ids.parquet", index=False
    )
    pd.DataFrame({
        "query_id": ["q"], "query_text": ["text"],
        "relevant_segment_id": ["capera:test__a.mp4:0.000:5.000"],
        "relevant_video_id": ["test__a.mp4"], "caption_index": [0],
        "caption_source": ["human"],
    }).to_parquet(tmp_path / "capera_query_ids.parquet", index=False)
    (tmp_path / "embedding_manifest.json").write_text(json.dumps({
        "item_count": 1, "query_count": 1, "split": "test",
        "embedding_mode": "real", "model_revision": "revision",
    }), encoding="utf-8")
    (tmp_path / "query_embeddings.json").write_text(json.dumps({
        "queries": {"demo": _unit_vectors(1)[0].tolist()},
    }), encoding="utf-8")

    result = validate_capera_artifacts(tmp_path, {
        "split": "test", "items": 1, "captions_per_item": 1, "queries": 1,
    })
    assert result["ok"] is False
    assert "provenance_unknown=False" in result["detail"]
