from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


DIMENSIONS = (2048, 1024, 512, 256)
VECTOR_BACKENDS = ("clickhouse", "qdrant", "pgvector")
EMBEDDING_MODES = {"real", "cached", "hybrid_text", "synthetic"}
FILTER_EXECUTION_MODES = {"pushdown", "legacy_candidate_ids"}
ATTENTION_IMPLEMENTATIONS = {"sdpa", "flash_attention_2"}


def capera_protocol() -> dict[str, object]:
    """Load the optional CapERA quality protocol only when it is requested."""
    candidates = (
        Path(os.getenv("PROJECT_CONFIG_PATH", "config.yaml")),
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path(__file__).resolve().parents[1] / "config.yaml",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("config.yaml not found for CapERA quality protocol")
    root = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        config = root["datasets"]["capera"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config.yaml has no datasets.capera quality protocol") from exc
    return {
        "split": str(config["quality_split"]),
        "items": int(config["quality_item_count"]),
        "captions_per_item": int(config["captions_per_item"]),
        "queries": int(config["quality_query_count"]),
    }


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    return int(env.get(name, str(default)))


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    return float(env.get(name, str(default)))


def _bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(env: Mapping[str, str], name: str, default: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip().lower() for part in env.get(name, default).split(",") if part.strip()))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _dimensions(env: Mapping[str, str]) -> tuple[int, ...]:
    raw = _csv(env, "ENABLED_DIMENSIONS", "512")
    try:
        values = tuple(int(value) for value in raw)
    except ValueError as exc:
        raise ValueError("ENABLED_DIMENSIONS must be a comma-separated integer list") from exc
    return values


@dataclass(frozen=True)
class Settings:
    embedding_mode: str
    enabled_vector_backends: tuple[str, ...]
    enabled_dimensions: tuple[int, ...]
    filter_execution_mode: str
    legacy_candidate_limit: int
    decode_chunk_s: float
    decode_prefetch_windows: int
    embed_batch_size: int
    db_write_batch_size: int
    media_max_clip_s: float
    media_cache_max_gb: float
    media_cache_retention_hours: float
    media_h264_crf: int
    require_secure_credentials: bool
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    ch_host: str
    ch_port: int
    ch_user: str
    ch_password: str
    ch_db: str
    qdrant_url: str
    milvus_uri: str
    artifacts_root: Path
    data_root: Path
    default_top_k: int
    bind_host: str
    qwen_repo_path: Path
    qwen_model_path: Path
    qwen_model_id: str
    qwen_model_revision: str
    qwen_source_commit: str
    model_bundle_root: Path
    attn_impl: str
    qwen_text_warm_limit_s: float
    qwen_text_warm_runs: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        instance = cls(
            embedding_mode=values.get("EMBEDDING_MODE", "synthetic").lower(),
            enabled_vector_backends=_csv(values, "ENABLED_VECTOR_BACKENDS", "clickhouse"),
            enabled_dimensions=_dimensions(values),
            filter_execution_mode=values.get("FILTER_EXECUTION_MODE", "pushdown").lower(),
            legacy_candidate_limit=_int(values, "LEGACY_CANDIDATE_LIMIT", 100000),
            decode_chunk_s=_float(values, "DECODE_CHUNK_S", 300.0),
            decode_prefetch_windows=_int(values, "DECODE_PREFETCH_WINDOWS", 8),
            embed_batch_size=_int(values, "EMBED_BATCH_SIZE", 2),
            db_write_batch_size=_int(values, "DB_WRITE_BATCH_SIZE", 512),
            media_max_clip_s=_float(values, "MEDIA_MAX_CLIP_S", 60.0),
            media_cache_max_gb=_float(values, "MEDIA_CACHE_MAX_GB", 10.0),
            media_cache_retention_hours=_float(values, "MEDIA_CACHE_RETENTION_HOURS", 168.0),
            media_h264_crf=_int(values, "MEDIA_H264_CRF", 23),
            require_secure_credentials=_bool(values, "REQUIRE_SECURE_CREDENTIALS"),
            pg_host=values.get("POSTGRES_HOST", "localhost"),
            pg_port=_int(values, "POSTGRES_PORT", 5442),
            pg_db=values.get("POSTGRES_DB", "uav_search"),
            pg_user=values.get("POSTGRES_USER", "uav"),
            pg_password=values.get("POSTGRES_PASSWORD", ""),
            ch_host=values.get("CLICKHOUSE_HOST", "localhost"),
            ch_port=_int(values, "CLICKHOUSE_PORT", 8143),
            ch_user=values.get("CLICKHOUSE_USER", "mvi"),
            ch_password=values.get("CLICKHOUSE_PASSWORD", ""),
            ch_db=values.get("CLICKHOUSE_DB", "uav_search"),
            qdrant_url=values.get("QDRANT_URL", "http://localhost:6343"),
            milvus_uri=values.get("MILVUS_URI", "http://localhost:19530"),
            artifacts_root=Path(values.get("ARTIFACTS_ROOT", "artifacts")),
            data_root=Path(values.get("DATA_ROOT", "data")),
            default_top_k=_int(values, "DEFAULT_TOP_K", 10),
            bind_host=values.get("BIND_HOST", "127.0.0.1"),
            qwen_repo_path=Path(values.get("QWEN_REPO_PATH", "/opt/Qwen3-VL-Embedding")),
            qwen_model_path=Path(values.get("QWEN_MODEL_PATH", "/models/Qwen3-VL-Embedding-2B")),
            qwen_model_id=values.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B"),
            qwen_model_revision=values.get("QWEN_MODEL_REVISION", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"),
            qwen_source_commit=values.get(
                "QWEN_SOURCE_COMMIT", "393e2978d27852b0d0230d6994f37f9c15bed73c"
            ),
            model_bundle_root=Path(values.get("MODEL_BUNDLE_ROOT", "/opt/mvi-model-bundle")),
            attn_impl=values.get("ATTN_IMPL", "sdpa").lower(),
            qwen_text_warm_limit_s=_float(values, "QWEN_TEXT_WARM_LIMIT_S", 10.0),
            qwen_text_warm_runs=_int(values, "QWEN_TEXT_WARM_RUNS", 3),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        if self.embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"EMBEDDING_MODE must be one of {sorted(EMBEDDING_MODES)}")
        known_backends = set(VECTOR_BACKENDS)
        unknown_backends = sorted(set(self.enabled_vector_backends) - known_backends)
        if unknown_backends:
            raise ValueError(f"unknown ENABLED_VECTOR_BACKENDS: {unknown_backends}")
        unknown_dimensions = sorted(set(self.enabled_dimensions) - set(DIMENSIONS))
        if unknown_dimensions:
            raise ValueError(f"unsupported ENABLED_DIMENSIONS: {unknown_dimensions}; maximum Qwen dimension is 2048")
        if self.filter_execution_mode not in FILTER_EXECUTION_MODES:
            raise ValueError(f"FILTER_EXECUTION_MODE must be one of {sorted(FILTER_EXECUTION_MODES)}")
        if self.attn_impl not in ATTENTION_IMPLEMENTATIONS:
            raise ValueError(f"ATTN_IMPL must be one of {sorted(ATTENTION_IMPLEMENTATIONS)}")
        if self.legacy_candidate_limit < 1:
            raise ValueError("LEGACY_CANDIDATE_LIMIT must be positive")
        if self.decode_chunk_s <= 0:
            raise ValueError("DECODE_CHUNK_S must be positive")
        if min(self.decode_prefetch_windows, self.embed_batch_size, self.db_write_batch_size) < 1:
            raise ValueError("decode prefetch, embed batch, and DB write batch sizes must be positive")
        if min(self.media_max_clip_s, self.media_cache_max_gb, self.media_cache_retention_hours) <= 0:
            raise ValueError("media clip/cache limits must be positive")
        if not 0 <= self.media_h264_crf <= 51:
            raise ValueError("MEDIA_H264_CRF must be in [0,51]")
        if self.require_secure_credentials:
            missing = []
            if not self.pg_password:
                missing.append("POSTGRES_PASSWORD")
            if not self.ch_password:
                missing.append("CLICKHOUSE_PASSWORD")
            if missing:
                raise ValueError(f"required credentials are empty: {', '.join(missing)}")

    @property
    def disabled_vector_backends(self) -> tuple[str, ...]:
        return tuple(name for name in VECTOR_BACKENDS if name not in self.enabled_vector_backends)


settings = Settings.from_env()
