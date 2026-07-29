from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np

from app.config import settings
from app.search.strategies import STRATEGIES

QUERIES=(
    "kalabalık trafik","otobüsleri göster","yürüyen insanlar","yüksek irtifa araçlar",
    "alçak uçuş","hızlı drone","gece görüntüsü","çok sayıda insan","boş yol","kırmızı araç",
    "park edilmiş arabalar","kavşakta trafik","bina yakınında insanlar","açık alanda araç",
    "keskin dönüş","eğik kamera","şehir merkezi","otoyolda otobüs","tek kişi","yoğun araç grubu",
)
MANDATORY_COLUMNS=(
    "block","embedding_mode","backend","backend_version","dimension","storage_type","dataset_id","corpus_size",
    "scale_level","real_unique_segments","physical_rows","replication_factor","filter_selectivity_target","filter_selectivity_actual",
    "strategy","pattern","returned_count","underfilled","filter_correctness","topk_agreement","ann_recall_vs_exact",
    "quality_vs_groundtruth","p50_ms","p95_ms","p99_ms","cold_ms","ingest_s","index_build_s","vector_storage_mb",
    "index_storage_mb","metadata_storage_mb","storage_amplification","index_params_json","settings_json","plan_used_vector_index",
    "indexed_vectors_count","max_limit_setting","hardware_profile",
)


def configurations():
    selectivities=(1.0,0.5,0.1,0.01,0.001)
    for backend in ("clickhouse","qdrant","pgvector"):
        strategies=STRATEGIES[backend]
        for dim in (2048,512):
            for strategy in strategies:
                for selectivity in selectivities: yield backend,strategy,dim,selectivity
        default="prefilter" if backend in ("clickhouse","qdrant") else "ann"
        for dim in (1024,256):
            for selectivity in selectivities: yield backend,default,dim,selectivity


def empty_row(backend,strategy,dimension,selectivity,dataset_id,corpus_size):
    row={key:None for key in MANDATORY_COLUMNS}
    row.update({"block":"C","embedding_mode":settings.embedding_mode,"backend":backend,"backend_version":"runtime",
        "dimension":dimension,"storage_type":"halfvec" if backend=="pgvector" and dimension==2048 else "float32",
        "dataset_id":dataset_id,"corpus_size":corpus_size,"scale_level":"L2","real_unique_segments":corpus_size,
        "physical_rows":corpus_size,"replication_factor":1,"filter_selectivity_target":selectivity,"strategy":strategy,
        "pattern":"A","quality_vs_groundtruth":None,"hardware_profile":f"{platform.system()}-{platform.machine()}-{platform.processor() or 'unknown'}",
        "index_params_json":json.dumps({"m":16,"ef_construction":128}),"settings_json":"{}"})
    return row


def run_configuration(api_url,backend,strategy,dimension,selectivity,dataset_id,corpus_size):
    row=empty_row(backend,strategy,dimension,selectivity,dataset_id,corpus_size); totals=[]; cold=None; last=None
    telemetry_filter={}
    thresholds_path=settings.artifacts_dir/"research/selectivity_thresholds_v2.json"
    if selectivity < 1.0 and thresholds_path.exists():
        thresholds=json.loads(thresholds_path.read_text(encoding="utf-8")); info=thresholds.get("altitude_m",{}).get("levels",{}).get(str(selectivity),{})
        if info.get("threshold") is not None: telemetry_filter={"altitude_m":[None,info["threshold"]]}
    for query in QUERIES:
        body={"query":query,"dataset_id":dataset_id,"backend":backend,"strategy":strategy,"dimension":dimension,
            "adaptive_mrl":{"enabled":False,"base_dim":256,"top_n":100},"metadata_filters":{},"telemetry_filters":telemetry_filter,"pattern":"A","top_k":10,"repeats":10}
        started=time.perf_counter()
        response=httpx.post(api_url+"/search",json=body,timeout=300); response.raise_for_status(); last=response.json()
        elapsed=(time.perf_counter()-started)*1000; cold=elapsed if cold is None else cold; totals.append(last["timings_stats"]["p50"])
    diag=last["diagnostics"]
    actual=diag.get("candidate_count",corpus_size)/corpus_size if corpus_size else None
    row.update({"filter_selectivity_actual":actual,"returned_count":diag["returned_count"],"underfilled":diag["underfilled"],
        "filter_correctness":diag["filter_correctness"],"topk_agreement":diag.get("topk_agreement"),"ann_recall_vs_exact":diag.get("ann_recall_at_k"),
        "p50_ms":float(np.percentile(totals,50)),"p95_ms":float(np.percentile(totals,95)),"p99_ms":float(np.percentile(totals,99)),
        "cold_ms":cold,"plan_used_vector_index":diag.get("plan_used_vector_index"),"indexed_vectors_count":diag.get("indexed_vectors_count"),
        "max_limit_setting":diag.get("max_limit_setting")})
    return row


def run(out: Path,api_url: str,dataset_id: str="auair") -> list[dict]:
    try:
        stats=httpx.get(api_url+"/stats",timeout=5).json(); corpus_size=next(x["segments"] for x in stats["datasets"] if x["dataset_id"]==dataset_id); offline_error=None
    except Exception as exc:
        corpus_size=0; offline_error=f"API unavailable: {type(exc).__name__}: {exc}"
    rows=[]
    for backend,strategy,dimension,selectivity in configurations():
        if offline_error:
            row=empty_row(backend,strategy,dimension,selectivity,dataset_id,corpus_size); row["settings_json"]=json.dumps({"benchmark_status":"blocked","error":offline_error})
        else:
            try: row=run_configuration(api_url,backend,strategy,dimension,selectivity,dataset_id,corpus_size)
            except Exception as exc:
                row=empty_row(backend,strategy,dimension,selectivity,dataset_id,corpus_size); row["settings_json"]=json.dumps({"benchmark_status":"error","error":str(exc)})
        rows.append(row)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=MANDATORY_COLUMNS); writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--level",default="L2",choices=["L2"]); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--api-url",default="http://localhost:8000"); parser.add_argument("--dataset",default="auair"); args=parser.parse_args()
    rows=run(args.out,args.api_url,args.dataset); print(json.dumps({"rows":len(rows),"embedding_mode":settings.embedding_mode,"out":str(args.out)}))


if __name__=="__main__": main()
