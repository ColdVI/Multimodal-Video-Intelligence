from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import FILTER_EXECUTION_MODES, settings
from app.db import postgres
from app.db.registry import enabled_adapters, enabled_health, initialize_enabled_backends
from app.embedding.router import mode_details
from app.search.strategies import SUPPORTED_STRATEGIES


@asynccontextmanager
async def lifespan(_: FastAPI):
    postgres.init_schema(include_vectors=False)
    initialize_enabled_backends()
    yield


app = FastAPI(title="Multimodal Video Intelligence", version="11.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


class AdaptiveMRL(BaseModel):
    enabled: bool = False
    base_dim: Literal[256, 512] = 256
    top_n: int = Field(default=100, ge=1, le=10000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    dataset_id: str
    backend: str = "clickhouse"
    strategy: str = "prefilter"
    dimension: Literal[2048, 1024, 512, 256] = 512
    adaptive_mrl: AdaptiveMRL = Field(default_factory=AdaptiveMRL)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    telemetry_filters: dict[str, Any] = Field(default_factory=dict)
    pattern: Literal["A", "B", "C"] = "A"
    top_k: int = Field(default=10, ge=1, le=10000)
    repeats: int = Field(default=1, ge=1, le=20)
    filter_execution_mode: Literal["pushdown", "legacy_candidate_ids"] | None = None


@app.get("/health")
def health(dataset_id: str | None = None) -> dict[str, Any]:
    metadata_ok = postgres.health()
    vector_health = enabled_health()
    response = {
        "status": "ok" if metadata_ok and all(vector_health.values()) else "degraded",
        "metadata": {"postgres": metadata_ok},
        "vector_backends": vector_health,
        "disabled_backends": list(settings.disabled_vector_backends),
        "enabled_dimensions": list(settings.enabled_dimensions),
        "filter_execution_mode": settings.filter_execution_mode,
        "pg": metadata_ok,
        "embedding_mode": settings.embedding_mode,
        "embedding": mode_details(dataset_id),
    }
    for name, ok in vector_health.items():
        response["ch" if name == "clickhouse" else name] = ok
    return response


@app.get("/stats")
def stats() -> dict[str, Any]:
    datasets = postgres.stats(
        dimensions=settings.enabled_dimensions,
        include_pgvector="pgvector" in settings.enabled_vector_backends,
    )
    for item in datasets:
        for adapter in enabled_adapters():
            if adapter.name == "pgvector":
                continue
            item[adapter.name] = {
                str(dimension): adapter.table_count(item["dataset_id"], dimension)
                for dimension in settings.enabled_dimensions
            }
        item["enabled_backends"] = list(settings.enabled_vector_backends)
        item["enabled_dimensions"] = list(settings.enabled_dimensions)
    return {
        "embedding_mode": settings.embedding_mode,
        "enabled_backends": list(settings.enabled_vector_backends),
        "enabled_dimensions": list(settings.enabled_dimensions),
        "datasets": datasets,
    }


@app.get("/facets/{dataset_id}")
def facets(dataset_id: str) -> dict[str, Any]:
    return {"dataset_id": dataset_id, **postgres.facets(dataset_id)}


@app.get("/readiness-data")
def readiness_data() -> dict[str, Any]:
    return {
        "pg_schema": postgres.schema_status(
            dimensions=settings.enabled_dimensions,
            include_vectors="pgvector" in settings.enabled_vector_backends,
        ),
        "capera_groundtruth": postgres.groundtruth_stats("capera"),
    }


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    active_strategies = {
        name: SUPPORTED_STRATEGIES[name]
        for name in settings.enabled_vector_backends
    }
    return {
        "enabled_backends": list(settings.enabled_vector_backends),
        "enabled_dimensions": list(settings.enabled_dimensions),
        "strategies": active_strategies,
        "dimensions": list(settings.enabled_dimensions),
        "filter_execution_mode": settings.filter_execution_mode,
        "filter_execution_modes": sorted(FILTER_EXECUTION_MODES),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        if request.backend not in settings.enabled_vector_backends:
            raise ValueError(
                f"backend {request.backend!r} is disabled; enabled backends: {settings.enabled_vector_backends}"
            )
        if request.dimension not in settings.enabled_dimensions:
            raise ValueError(
                f"dimension {request.dimension} is disabled; enabled dimensions: {settings.enabled_dimensions}"
            )
        from app.search.engine import search as run_search

        return run_search(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
