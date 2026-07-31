from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import FILTER_EXECUTION_MODES, settings
from app.auth import TokenAuthMiddleware
from app.db import postgres
from app.db.telemetry_registry import fields_for_run
from app.db.registry import enabled_adapters, enabled_health, initialize_enabled_backends
from app.embedding.router import mode_details
from app.search.filter_schema import CANONICAL_FILTER_FIELDS, serialize_fields
from app.search.strategies import SUPPORTED_STRATEGIES


@asynccontextmanager
async def lifespan(_: FastAPI):
    postgres.init_schema(include_vectors=False)
    initialize_enabled_backends()
    yield


app = FastAPI(title="Multimodal Video Intelligence", version="11.0.0", lifespan=lifespan)
app.add_middleware(TokenAuthMiddleware, token=settings.api_token)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


ADAPTIVE_MRL_ALLOWED_PAIRS: frozenset[tuple[int, int]] = frozenset({
    (256, 512), (256, 1024), (256, 2048), (512, 1024), (512, 2048),
})


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
    diagnose: bool = False
    explain: bool = False


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
                str(dimension): (
                    adapter.count_run(item["dataset_id"], item["active_run_id"], dimension)
                    if item.get("active_run_id") else adapter.table_count(item["dataset_id"], dimension)
                )
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


@app.get("/datasets")
def datasets() -> dict[str, Any]:
    return {"datasets": postgres.list_datasets()}


@app.get("/datasets/{dataset_id}/runs")
def dataset_runs(dataset_id: str) -> dict[str, Any]:
    return {"dataset_id": dataset_id, "runs": postgres.list_runs(dataset_id)}


@app.get("/datasets/{dataset_id}/filter-schema")
def filter_schema(dataset_id: str) -> dict[str, Any]:
    snapshot = postgres.get_active_run_snapshot(dataset_id)
    facet_data = postgres.facets(dataset_id)
    if snapshot is None:
        names = {"event_category", "split", "video_id"}
        names.update(name for name, bounds in facet_data.get("telemetry", {}).items() if bounds)
        names.update(name for name, bounds in facet_data.get("counts", {}).items() if bounds)
        fields = serialize_fields({name: CANONICAL_FILTER_FIELDS[name] for name in names})
        run_id = None
    else:
        run_id = str(snapshot["run_id"])
        fields = fields_for_run(dataset_id, run_id)
        registered = {field["name"] for field in fields}
        for name in ("event_category", "split", "video_id", "person_count", "vehicle_count", "bus_count", "is_night"):
            available = (
                name in {"event_category", "split", "video_id"}
                or facet_data.get("counts", {}).get(name) is not None
                or (name == "is_night" and bool(facet_data.get("booleans", {}).get(name)))
            )
            if available and name not in registered:
                fields.extend(serialize_fields({name: CANONICAL_FILTER_FIELDS[name]}))
    for field in fields:
        name = field["name"]
        field["bounds"] = (
            facet_data.get("telemetry", {}).get(name)
            or facet_data.get("counts", {}).get(name)
        )
        field["values"] = (
            facet_data.get("event_categories") if name == "event_category" else
            facet_data.get("splits") if name == "split" else
            facet_data.get("video_ids") if name == "video_id" else
            facet_data.get("booleans", {}).get(name)
        )
        field["wrap"] = field.get("data_type") == "circular_deg"
    return {"dataset_id": dataset_id, "run_id": run_id, "fields": fields, "extra_filterable": False}


@app.get("/ingest-runs/{run_id}")
def ingest_run(run_id: str) -> dict[str, Any]:
    result = postgres.run_info(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="ingest run not found")
    return result


@app.get("/media/{segment_id}/info")
def media_information(segment_id: str, run_id: str | None = None) -> dict[str, Any]:
    from app.media import media_info

    return media_info(segment_id, run_id)


@app.get("/media/{segment_id}", response_class=FileResponse)
def media(segment_id: str, run_id: str | None = None):
    from app.media import MediaError, get_clip

    try:
        path, _ = get_clip(segment_id, run_id)
    except MediaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc
    return FileResponse(path, media_type="video/mp4", filename=f"{segment_id}.mp4")


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
        if request.adaptive_mrl.enabled:
            base_dim = request.adaptive_mrl.base_dim
            if base_dim not in settings.enabled_dimensions:
                raise ValueError(
                    f"adaptive_mrl.base_dim {base_dim} is disabled; enabled dimensions: {settings.enabled_dimensions}"
                )
            if base_dim >= request.dimension:
                raise ValueError(
                    f"adaptive_mrl.base_dim {base_dim} must be smaller than dimension {request.dimension}"
                )
            if (base_dim, request.dimension) not in ADAPTIVE_MRL_ALLOWED_PAIRS:
                raise ValueError(
                    f"adaptive_mrl pair (base_dim={base_dim}, dimension={request.dimension}) is not in the "
                    f"supported allow-list {sorted(ADAPTIVE_MRL_ALLOWED_PAIRS)}"
                )
            if request.adaptive_mrl.top_n < request.top_k:
                raise ValueError(
                    f"adaptive_mrl.top_n {request.adaptive_mrl.top_n} must be >= top_k {request.top_k}"
                )
        from app.search.engine import search as run_search

        return run_search(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
