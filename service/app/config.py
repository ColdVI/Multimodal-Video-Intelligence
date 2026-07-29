from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    embedding_mode: str = _env("EMBEDDING_MODE", "synthetic")
    embedding_dim: int = int(_env("EMBEDDING_DIM", "2048"))
    artifacts_dir: Path = Path(_env("ARTIFACTS_DIR", "/workspace/artifacts"))
    pg_dsn: str = _env("PG_DSN", "postgresql://faz7:faz7@pg:5432/faz7")
    ch_host: str = _env("CH_HOST", "ch")
    ch_port: int = int(_env("CH_PORT", "8123"))
    ch_user: str = _env("CH_USER", "default")
    ch_password: str = _env("CH_PASSWORD", "")
    ch_database: str = _env("CH_DATABASE", "faz7")
    qdrant_url: str = _env("QDRANT_URL", "http://qdrant:6333")
    api_url: str = _env("API_URL", "http://api:8000")
    qwen_model: str = _env("QWEN_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
    qwen_repo: Path = Path(_env("QWEN_REPO", "/opt/Qwen3-VL-Embedding"))
    torch_dtype: str = _env("TORCH_DTYPE", "float16")

    def __post_init__(self) -> None:
        if self.embedding_mode not in {"real", "cached", "synthetic"}:
            raise ValueError("EMBEDDING_MODE must be real, cached, or synthetic")


settings = Settings()
