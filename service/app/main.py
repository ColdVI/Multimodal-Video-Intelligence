from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import DIMENSIONS, settings
from app.db import clickhouse, postgres, qdrant
from app.embedding.router import mode_details
from app.search.strategies import SUPPORTED_STRATEGIES


@asynccontextmanager
async def lifespan(_: FastAPI):
    postgres.init_schema()
    clickhouse.init_schema()
    qdrant.init_schema()
    yield


app = FastAPI(title="Multimodal Video Intelligence — Faz 7", version="7.0.0", lifespan=lifespan)


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


@app.get("/health")
def health(dataset_id: str | None = None) -> dict[str, Any]:
    return {
        "status": "ok" if all((postgres.health(), clickhouse.health(), qdrant.health())) else "degraded",
        "pg": postgres.health(),
        "ch": clickhouse.health(),
        "qdrant": qdrant.health(),
        "embedding_mode": settings.embedding_mode,
        "embedding": mode_details(dataset_id),
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    datasets = postgres.stats()
    for item in datasets:
        item["clickhouse"] = {str(d): clickhouse.table_count(item["dataset_id"], d) for d in DIMENSIONS}
        item["qdrant"] = {str(d): qdrant.table_count(item["dataset_id"], d) for d in DIMENSIONS}
    return {"embedding_mode": settings.embedding_mode, "datasets": datasets}


@app.get("/facets/{dataset_id}")
def facets(dataset_id: str) -> dict[str, Any]:
    return {"dataset_id": dataset_id, **postgres.facets(dataset_id)}


@app.get("/readiness-data")
def readiness_data() -> dict[str, Any]:
    return {
        "pg_schema": postgres.schema_status(),
        "capera_groundtruth": postgres.groundtruth_stats("capera"),
    }


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    return {"strategies": SUPPORTED_STRATEGIES, "dimensions": DIMENSIONS}


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        from app.search.engine import search as run_search

        return run_search(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
