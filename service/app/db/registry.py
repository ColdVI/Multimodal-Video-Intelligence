from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from app.config import settings


@dataclass(frozen=True)
class BackendAdapter:
    name: str
    module_name: str
    init_schema: Callable[[tuple[int, ...]], None]
    health: Callable[[], bool]
    table_count: Callable[[str, int], int]
    write_chunk: Callable[..., int]
    delete_inactive_chunk: Callable[[str, str, int, int], int]
    count_run: Callable[[str, str, int], int]
    search: Callable[..., Any]
    delete_run: Callable[[str, str, int], int]


def _module(name: str) -> Any:
    target = "postgres" if name == "pgvector" else name
    return import_module(f"app.db.{target}")


def _adapter(name: str) -> BackendAdapter:
    def init_schema(dimensions: tuple[int, ...]) -> None:
        module = _module(name)
        if name == "pgvector":
            module.init_schema(dimensions=dimensions, include_vectors=True)
        else:
            module.init_schema(dimensions=dimensions)

    return BackendAdapter(
        name=name,
        module_name="postgres" if name == "pgvector" else name,
        init_schema=init_schema,
        health=lambda: bool(_module(name).health()),
        table_count=lambda dataset_id, dimension: int(_module(name).table_count(dataset_id, dimension)),
        write_chunk=(
            lambda run_id, dataset_id, dimension, chunk_index, rows:
                _module(name).write_run_vectors(
                    run_id, dataset_id, dimension, chunk_index,
                    [(row["segment_id"], row["embedding"]) for row in rows],
                )
            if name == "pgvector" else
            lambda run_id, dataset_id, dimension, chunk_index, rows:
                _module(name).write_run_chunk(run_id, dataset_id, dimension, chunk_index, rows)
        ),
        delete_inactive_chunk=lambda run_id, dataset_id, chunk_index, dimension: int(
            _module(name).delete_inactive_chunk(run_id, dataset_id, chunk_index, dimension)
        ),
        count_run=lambda dataset_id, run_id, dimension: int(
            _module(name).count_run(dataset_id, run_id, dimension)
        ),
        search=lambda *args, **kwargs: _module(name).search_vectors(*args, **kwargs),
        delete_run=lambda dataset_id, run_id, dimension: int(
            _module(name).delete_run(dataset_id, run_id, dimension)
        ),
    )


BACKEND_REGISTRY = {
    name: _adapter(name)
    for name in ("clickhouse", "qdrant", "pgvector")
}


def enabled_adapters() -> tuple[BackendAdapter, ...]:
    return tuple(BACKEND_REGISTRY[name] for name in settings.enabled_vector_backends)


def initialize_enabled_backends() -> None:
    for adapter in enabled_adapters():
        adapter.init_schema(settings.enabled_dimensions)


def enabled_health() -> dict[str, bool]:
    return {adapter.name: adapter.health() for adapter in enabled_adapters()}


__all__ = [
    "BACKEND_REGISTRY",
    "BackendAdapter",
    "enabled_adapters",
    "enabled_health",
    "initialize_enabled_backends",
]
