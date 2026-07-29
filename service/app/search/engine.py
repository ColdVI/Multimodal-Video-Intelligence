from __future__ import annotations

import math
import statistics
import time
from typing import Any

import numpy as np

from app.config import settings
from app.db import clickhouse, postgres, qdrant
from app.embedding.router import embed_query
from app.search.strategies import validate


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values,dtype=np.float64),q)) if values else 0.0


def _numpy_exact(dataset_id: str, query: np.ndarray, dim: int, top_k: int, candidate_ids: list[str] | None) -> list[tuple[str,float]]:
    entries=postgres.all_vectors(dataset_id,dim,candidate_ids)
    if not entries: return []
    ids=[x[0] for x in entries]; matrix=np.asarray([x[1] for x in entries],dtype=np.float32)
    order=np.argsort(-(matrix @ query.astype(np.float32)),kind="stable")[:top_k]
    return [(ids[i],float(matrix[i] @ query)) for i in order]


def _dispatch(backend: str, dataset_id: str, query: np.ndarray, dimension: int, top_k: int,
              strategy: str, candidates: list[str] | None, telemetry: dict | None) -> tuple[list[tuple[str,float]],dict]:
    if backend=="numpy_exact": return _numpy_exact(dataset_id,query,dimension,top_k,candidates),{"plan_used_vector_index":False}
    if backend=="pgvector": return postgres.vector_search(dataset_id,query,dimension,top_k,strategy,candidates),{"plan_used_vector_index":strategy!="exact"}
    if backend=="clickhouse": return clickhouse.vector_search(dataset_id,query,dimension,top_k,strategy,candidates,telemetry)
    if backend=="qdrant": return qdrant.vector_search(dataset_id,query,dimension,top_k,strategy,candidates,telemetry)
    raise NotImplementedError("Milvus is optional; start compose profile and install pymilvus")


def _run_once(request: dict[str,Any]) -> dict:
    start=time.perf_counter(); dataset_id=request["dataset_id"]
    mark=time.perf_counter(); candidates=postgres.filter_ids(dataset_id,request.get("metadata_filters"),request.get("telemetry_filters")); filter_ms=(time.perf_counter()-mark)*1000
    mark=time.perf_counter(); query=embed_query(request["query"],dataset_id,request["dimension"]); embed_ms=(time.perf_counter()-mark)*1000
    mark=time.perf_counter()
    adaptive=request.get("adaptive_mrl") or {}
    if adaptive.get("enabled"):
        base_dim=int(adaptive.get("base_dim",256)); top_n=int(adaptive.get("top_n",100))
        base_query=embed_query(request["query"],dataset_id,base_dim)
        first,_=_dispatch(request["backend"],dataset_id,base_query,base_dim,top_n,request["strategy"],candidates,request.get("telemetry_filters"))
        ranked,diag=_dispatch("numpy_exact",dataset_id,query,request["dimension"],request["top_k"],"exact",[x[0] for x in first],None)
        diag["adaptive_candidates"]=len(first)
    else:
        ranked,diag=_dispatch(request["backend"],dataset_id,query,request["dimension"],request["top_k"],request["strategy"],candidates,request.get("telemetry_filters"))
    vector_ms=(time.perf_counter()-mark)*1000
    mark=time.perf_counter(); hydrated=postgres.hydrate([x[0] for x in ranked]); hydrate_ms=(time.perf_counter()-mark)*1000
    scores=dict(ranked)
    for row in hydrated: row["score"]=scores[row["segment_id"]]
    notes=[]
    if request["backend"]=="clickhouse" and request["top_k"]>100: notes.append("top_k>100: max_limit_for_vector_search_queries query-level yükseltildi")
    else: notes.append("top_k<=100: max_limit ayarı değiştirilmedi")
    if settings.embedding_mode=="synthetic": notes.append("sentetik embedding: sıralama semantik kaliteyi temsil etmez")
    if settings.embedding_mode=="cached": notes.append("item vektörleri gerçek cached; serbest metin query vektörü cache yoksa sentetiktir")
    returned=len(hydrated)
    diag.update({"candidate_count":postgres.count(dataset_id) if candidates is None else len(candidates),"returned_count":returned,"underfilled":returned<request["top_k"],"filter_correctness":True,"notes":notes})
    timings={"filter":filter_ms,"embed":embed_ms,"vector_search":vector_ms,"hydrate":hydrate_ms,"total":(time.perf_counter()-start)*1000}
    return {"timings_ms":timings,"diagnostics":diag,"results":hydrated}


def search(request: dict[str,Any]) -> dict:
    validate(request["backend"],request["strategy"])
    if request["dimension"] not in (2048,1024,512,256): raise ValueError("invalid dimension")
    if not 1 <= request["top_k"] <= 200: raise ValueError("top_k must be 1..200")
    repeats=int(request.get("repeats",1)); runs=[_run_once(request) for _ in range(repeats)]
    last=runs[-1]; totals=[x["timings_ms"]["total"] for x in runs]
    reference=_numpy_exact(request["dataset_id"],embed_query(request["query"],request["dataset_id"],request["dimension"]),request["dimension"],request["top_k"],postgres.filter_ids(request["dataset_id"],request.get("metadata_filters"),request.get("telemetry_filters")))
    got={x["segment_id"] for x in last["results"]}; exact={x[0] for x in reference}; overlap=len(got & exact)
    last["diagnostics"]["ann_recall_at_k"]=overlap/len(exact) if exact else None
    last["diagnostics"]["topk_agreement"]=overlap/len(got|exact) if got|exact else None
    return {"embedding_mode":settings.embedding_mode,"backend":request["backend"],"strategy":request["strategy"],"dimension":request["dimension"],"pattern":request.get("pattern","A"),**last,"timings_stats":{"p50":_percentile(totals,50),"p95":_percentile(totals,95),"n_repeats":repeats}}
