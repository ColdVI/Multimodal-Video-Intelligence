from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DIMENSIONS = (2048, 1024, 512, 256)
EMBEDDING_MODES = {"real", "cached", "synthetic"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    embedding_mode: str = os.getenv("EMBEDDING_MODE", "synthetic").lower()
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = _int("POSTGRES_PORT", 5442)
    pg_db: str = os.getenv("POSTGRES_DB", "uav_search")
    pg_user: str = os.getenv("POSTGRES_USER", "uav")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "uav_local_only")
    ch_host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    ch_port: int = _int("CLICKHOUSE_PORT", 8143)
    ch_user: str = os.getenv("CLICKHOUSE_USER", "default")
    ch_password: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    ch_db: str = os.getenv("CLICKHOUSE_DB", "uav_search")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6343")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    artifacts_root: Path = Path(os.getenv("ARTIFACTS_ROOT", "artifacts"))
    data_root: Path = Path(os.getenv("DATA_ROOT", "data"))
    default_top_k: int = _int("DEFAULT_TOP_K", 10)
    qwen_repo_path: Path = Path(os.getenv("QWEN_REPO_PATH", "/opt/Qwen3-VL-Embedding"))
    qwen_model_path: Path = Path(os.getenv("QWEN_MODEL_PATH", "/opt/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B"))

    def validate(self) -> None:
        if self.embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"EMBEDDING_MODE must be one of {sorted(EMBEDDING_MODES)}")


settings = Settings()
settings.validate()

