from __future__ import annotations

from contextlib import asynccontextmanager

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db import clickhouse, postgres, qdrant
from app.search.engine import search as run_search
from app.search.strategies import STRATEGIES


@asynccontextmanager
async def lifespan(_: FastAPI):
    postgres.init_schema()
    clickhouse.init_schema()
    qdrant.init_schema()
    yield


app = FastAPI(title="Multimodal Video Intelligence — Faz 7", version="7.0.0", lifespan=lifespan)


class AdaptiveMRL(BaseModel):
    enabled: bool = False
    base_dim: Literal[256,512] = 256
    top_n: int = Field(100,ge=1,le=200)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    dataset_id: str
    backend: Literal["clickhouse","qdrant","pgvector","numpy_exact","milvus"] = "clickhouse"
    strategy: str = "prefilter"
    dimension: Literal[2048,1024,512,256] = 512
    adaptive_mrl: AdaptiveMRL = AdaptiveMRL()
    metadata_filters: dict[str,Any] = {}
    telemetry_filters: dict[str,Any] = {}
    pattern: Literal["A","B","C"] = "A"
    top_k: int = Field(10,ge=1,le=200)
    repeats: int = Field(1,ge=1,le=20)


@app.get("/health")
def health() -> dict:
    state = {"status":"ok", "pg":postgres.healthy(), "ch":clickhouse.healthy(), "qdrant":qdrant.healthy(), "embedding_mode":settings.embedding_mode}
    if not all(state[key] for key in ("pg","ch","qdrant")): state["status"]="degraded"
    return state


@app.get("/stats")
def stats() -> dict:
    datasets=[]
    for item in postgres.stats():
        dataset_id=item["dataset_id"]
        item["backends"]={"pgvector":postgres.count(dataset_id),"clickhouse":clickhouse.count(dataset_id),"qdrant":qdrant.count(dataset_id)}
        item["consistent"]=all(value==item["segments"] for value in item["backends"].values())
        datasets.append(item)
    return {"embedding_mode":settings.embedding_mode,"datasets":datasets}


@app.get("/facets/{dataset_id}")
def facets(dataset_id: str) -> dict:
    if not any(x["dataset_id"]==dataset_id for x in postgres.stats()): raise HTTPException(404,"dataset not found")
    return postgres.facets(dataset_id)


@app.get("/strategies")
def strategies() -> dict:
    return {key:list(value) for key,value in STRATEGIES.items()}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    try:
        return run_search(request.model_dump())
    except (ValueError,NotImplementedError,FileNotFoundError) as exc:
        raise HTTPException(422,str(exc)) from exc
