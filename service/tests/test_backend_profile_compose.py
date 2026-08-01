from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _compose(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_qdrant_only_compose_has_no_clickhouse_dependency_or_service():
    compose = _compose("docker-compose.qdrant.yml")
    assert set(compose["services"]) == {"pg", "qdrant", "api", "ui"}
    assert set(compose["services"]["api"]["depends_on"]) == {"pg", "qdrant"}
    env = compose["services"]["api"]["environment"]
    assert env["ENABLED_VECTOR_BACKENDS"] == "qdrant"
    assert env["DEFAULT_VECTOR_BACKEND"] == "qdrant"


def test_pgvector_only_compose_has_no_clickhouse_or_qdrant():
    compose = _compose("docker-compose.pgvector.yml")
    assert set(compose["services"]) == {"pg", "api", "ui"}
    assert set(compose["services"]["api"]["depends_on"]) == {"pg"}
    env = compose["services"]["api"]["environment"]
    assert env["ENABLED_VECTOR_BACKENDS"] == "pgvector"
    assert env["DEFAULT_VECTOR_BACKEND"] == "pgvector"


def test_benchmark_backend_list_is_configurable_not_hardcoded():
    raw = (ROOT / "docker-compose.benchmark.yml").read_text(encoding="utf-8")
    assert "${BENCHMARK_VECTOR_BACKENDS:-clickhouse,qdrant,pgvector}" in raw
