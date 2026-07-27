import copy

from scripts.colab_gpu_bench import (
    QWEN_TRUNCATE_DIMS,
    SCHEMA_VERSION,
    _model_dtype_and_attn,
    migrate_legacy_schema,
)

LEGACY_V1 = {
    "hardware_profile": "colab_t4",
    "gpu_name": "Tesla T4",
    "embedding_speed": [
        {"model": "xclip_hf_zeroshot", "timing": {"embed_video": {"mean_s": 1.31}}},
        {
            "model": "qwen3vl_emb_2048",
            "timing": {"embed_video": {"mean_s": 341.42}},
        },
        {
            "model": "qwen3vl_emb_1024",
            "timing": {"embed_video": {"mean_s": 341.42}},
            "note": "qwen3vl_emb_2048 ile ayni olculmus timing",
        },
        {
            "model": "qwen3vl_emb_512",
            "timing": {"embed_video": {"mean_s": 341.42}},
            "note": "qwen3vl_emb_2048 ile ayni olculmus timing",
        },
        {
            "model": "qwen3vl_emb_256",
            "timing": {"embed_video": {"mean_s": 341.42}},
            "note": "qwen3vl_emb_2048 ile ayni olculmus timing",
        },
    ],
    "detector_speed": [],
    "frame_read_total_s": 1080.92,
}


def test_migrate_legacy_schema_merges_four_qwen_rows_into_one():
    migrated = migrate_legacy_schema(copy.deepcopy(LEGACY_V1))
    models = [r["model"] for r in migrated["embedding_speed"]]
    assert models == ["xclip_hf_zeroshot", "qwen3vl_emb"]

    qwen = migrated["embedding_speed"][1]
    assert qwen["truncate_dims"] == QWEN_TRUNCATE_DIMS
    assert qwen["timing"]["embed_video"]["mean_s"] == 341.42
    assert "note" not in qwen or "qwen3vl_emb_2048" not in qwen.get("model", "")


def test_migrate_legacy_schema_sets_version_and_is_idempotent():
    migrated = migrate_legacy_schema(copy.deepcopy(LEGACY_V1))
    assert migrated["schema_version"] == SCHEMA_VERSION
    twice = migrate_legacy_schema(migrated)
    assert twice is migrated  # zaten v2 ise oldugu gibi doner, tekrar isleme girmez


def test_migrate_legacy_schema_preserves_non_qwen_fields():
    migrated = migrate_legacy_schema(copy.deepcopy(LEGACY_V1))
    assert migrated["hardware_profile"] == "colab_t4"
    assert migrated["gpu_name"] == "Tesla T4"
    assert migrated["frame_read_total_s"] == 1080.92


def test_migrate_legacy_schema_handles_qwen_load_error_without_dropping_it():
    legacy = copy.deepcopy(LEGACY_V1)
    legacy["embedding_speed"] = [{"model": "qwen3vl_emb_2048", "error": "load basarisiz: oom"}]
    migrated = migrate_legacy_schema(legacy)
    # base (2048) hata verdiyse hata kaydi kaybolmamali
    assert any(r.get("model", "").startswith("qwen3vl_emb") for r in migrated["embedding_speed"])


def test_model_dtype_and_attn_missing_model_attribute_returns_none():
    class NoModel:
        pass

    info = _model_dtype_and_attn(NoModel())
    assert info == {"dtype": None, "attn_implementation": None}


def test_model_dtype_and_attn_reads_real_torch_module():
    import torch

    class FakeEmbedder:
        model = torch.nn.Linear(2, 2).to(torch.float32)

    info = _model_dtype_and_attn(FakeEmbedder())
    assert info["dtype"] == "torch.float32"
