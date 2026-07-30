from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


DIMENSIONS = (2048, 1024, 512, 256)
EMBEDDING_MODES = {"real", "cached", "hybrid_text", "synthetic"}


def _capera_protocol() -> dict[str, object]:
    candidates = (
        Path(os.getenv("PROJECT_CONFIG_PATH", "config.yaml")),
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path(__file__).resolve().parents[1] / "config.yaml",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("config.yaml not found for CapERA quality protocol")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["datasets"]["capera"]
    return {
        "split": str(config["quality_split"]),
        "items": int(config["quality_item_count"]),
        "captions_per_item": int(config["captions_per_item"]),
        "queries": int(config["quality_query_count"]),
    }


CAPERA_PROTOCOL = _capera_protocol()


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
    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B")
    qwen_model_revision: str = os.getenv("QWEN_MODEL_REVISION", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda")
    qwen_text_warm_limit_s: float = float(os.getenv("QWEN_TEXT_WARM_LIMIT_S", "10"))
    qwen_text_warm_runs: int = _int("QWEN_TEXT_WARM_RUNS", 3)

    def validate(self) -> None:
        if self.embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"EMBEDDING_MODE must be one of {sorted(EMBEDDING_MODES)}")


settings = Settings()
settings.validate()
