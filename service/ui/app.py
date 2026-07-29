from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

API_URL=os.getenv("API_URL","http://localhost:8000")
MODE=os.getenv("EMBEDDING_MODE","synthetic")


def api(method: str,path: str,**kwargs):
    response=httpx.request(method,API_URL+path,timeout=180,**kwargs); response.raise_for_status(); return response.json()


def banner(dataset_id: str | None=None) -> str:
    if MODE=="synthetic":
        return '<div class="mode-banner danger">SENTETİK EMBEDDING — sonuç sıralamaları anlamsızdır. Yalnızca sistem/gecikme doğrulaması.</div>'
    if MODE=="cached":
        count="?"
        if dataset_id:
            try: count=next(x["segments"] for x in api("GET","/stats")["datasets"] if x["dataset_id"]==dataset_id)
            except Exception: pass
        return f'<div class="mode-banner success">GERÇEK EMBEDDING (Qwen3-VL-2B, cached) — {dataset_id or "dataset"}: {count} vektör</div>'
    return f'<div class="mode-banner success">GERÇEK EMBEDDING — Qwen3-VL-2B / {os.getenv("TORCH_DTYPE","float16")} / {os.getenv("GPU_NAME","GPU runtime")}</div>'


def initial_data():
    try:
        stats=api("GET","/stats"); datasets=[x["dataset_id"] for x in stats["datasets"]]
    except Exception: datasets=[]
    try: strategies=api("GET","/strategies")
    except Exception: strategies={"clickhouse":["exact","ann","prefilter","postfilter"]}
    return datasets,strategies


DATASETS,STRATEGIES=initial_data()


def dataset_change(dataset_id: str):
    try: data=api("GET",f"/facets/{dataset_id}")
    except Exception: data={"event_categories":[],"splits":[],"video_ids":[],"telemetry":{}}
    t=data.get("telemetry",{}); has=any(value is not None for value in t.values())
    def range_update(lo,hi):
        lower,upper=lo or 0,hi or 1
        return gr.update(minimum=lower,maximum=upper,value=lower,visible=has),gr.update(minimum=lower,maximum=upper,value=upper,visible=has)
    alt=range_update(t.get("altitude_min"),t.get("altitude_max")); vel=range_update(t.get("velocity_min"),t.get("velocity_max")); gim=range_update(t.get("gimbal_pitch_min"),t.get("gimbal_pitch_max"))
    return banner(dataset_id),gr.update(choices=data.get("event_categories") or [],value=None),gr.update(choices=data.get("splits") or [],value=None),gr.update(choices=data.get("video_ids") or [],value=None),*alt,*vel,*gim,gr.update(visible=has)


def strategy_change(backend: str): return gr.update(choices=STRATEGIES.get(backend,[]),value=STRATEGIES.get(backend,["exact"])[0])


def payload(query,dataset,backend,strategy,dimension,adaptive,base_dim,top_n,pattern,top_k,repeats,event,split,video,altitude_min,altitude_max,velocity_min,velocity_max,gimbal_min,gimbal_max):
    meta={k:v for k,v in {"event_category":event,"split":split,"video_id":video}.items() if v not in (None,"")}
    telemetry={}
    for key,lo,hi in (("altitude_m",altitude_min,altitude_max),("velocity_mps",velocity_min,velocity_max),("gimbal_pitch",gimbal_min,gimbal_max)):
        if lo is not None or hi is not None: telemetry[key]=[lo,hi]
    return {"query":query,"dataset_id":dataset,"backend":backend,"strategy":strategy,"dimension":int(dimension),"adaptive_mrl":{"enabled":adaptive,"base_dim":int(base_dim),"top_n":int(top_n)},"metadata_filters":meta,"telemetry_filters":telemetry,"pattern":pattern,"top_k":int(top_k),"repeats":int(repeats)}


def run_search(*args):
    data=api("POST","/search",json=payload(*args))
    timing={**data["timings_ms"],**{f"stats_{k}":v for k,v in data["timings_stats"].items()}}
    rows=data["results"]
    fd,path=tempfile.mkstemp(prefix="faz7_results_",suffix=".csv"); os.close(fd)
    fields=["video_id","t_start","t_end","score","caption","file_path","altitude_m","velocity_mps","gimbal_pitch"]
    with open(path,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    table=[[r.get(x) for x in fields] for r in rows]
    return timing,data["diagnostics"],table,path


def compare(query,dataset,backends,dimensions,top_k):
    rows=[]
    for backend in backends:
        for strategy in STRATEGIES.get(backend,[]):
            for dimension in dimensions:
                try:
                    body=payload(query,dataset,backend,strategy,dimension,False,256,100,"A",top_k,10,None,None,None,None,None,None,None,None,None)
                    data=api("POST","/search",json=body); d=data["diagnostics"]
                    rows.append([backend,strategy,dimension,data["timings_stats"]["p50"],data["timings_stats"]["p95"],d.get("ann_recall_at_k"),d["returned_count"],d["underfilled"]])
                except Exception as exc: rows.append([backend,strategy,dimension,None,None,None,0,True,str(exc)])
    return rows


CSS="""
.mode-banner {padding:16px;border-radius:8px;font-weight:800;text-align:center;position:sticky;top:0;z-index:99}
.danger {background:#991b1b;color:white}.success {background:#166534;color:white}
"""
with gr.Blocks(title="Multimodal Video Intelligence",css=CSS) as demo:
    banner_box=gr.HTML(banner(DATASETS[0] if DATASETS else None))
    with gr.Tab("Ara"):
        with gr.Row():
            with gr.Column(scale=1):
                dataset=gr.Dropdown(DATASETS,value=DATASETS[0] if DATASETS else None,label="Dataset")
                with gr.Accordion("Metadata filtreleri"):
                    event=gr.Dropdown(label="Event category"); split=gr.Dropdown(label="Split"); video=gr.Dropdown(label="Video ID")
                with gr.Accordion("Telemetri filtreleri") as telemetry_panel:
                    with gr.Row(): altitude_min=gr.Slider(0,1,label="İrtifa min (m)"); altitude_max=gr.Slider(0,1,value=1,label="İrtifa max (m)")
                    with gr.Row(): velocity_min=gr.Slider(0,1,label="Hız min (m/s)"); velocity_max=gr.Slider(0,1,value=1,label="Hız max (m/s)")
                    with gr.Row(): gimbal_min=gr.Slider(0,1,label="Gimbal pitch min"); gimbal_max=gr.Slider(0,1,value=1,label="Gimbal pitch max")
                with gr.Accordion("Arama yöntemi",open=True):
                    backend=gr.Radio(["clickhouse","qdrant","pgvector","numpy_exact"],value="clickhouse",label="Backend")
                    strategy=gr.Dropdown(STRATEGIES.get("clickhouse",[]),value="prefilter",label="Strategy")
                    dimension=gr.Radio([2048,1024,512,256],value=512,label="Boyut")
                    adaptive=gr.Checkbox(False,label="Adaptive MRL"); base_dim=gr.Radio([256,512],value=256,label="Base dim"); top_n=gr.Slider(20,200,100,step=10,label="top_N")
                    pattern=gr.Radio(["A","B","C"],value="A",label="Pattern")
                top_k=gr.Slider(1,50,10,step=1,label="top_k"); repeats=gr.Slider(1,20,1,step=1,label="Tekrar")
            with gr.Column(scale=2):
                query=gr.Textbox(label="Serbest metin sorgusu",value="kalabalık trafik")
                search_button=gr.Button("Search",variant="primary")
                timing=gr.JSON(label="Gecikme paneli (ms)"); diagnostics=gr.JSON(label="Diagnostics")
                results=gr.Dataframe(headers=["video_id","t_start","t_end","score","caption","file_path","altitude_m","velocity_mps","gimbal_pitch"],label="Sonuçlar")
                download=gr.File(label="Sonucu CSV indir")
        dataset.change(dataset_change,dataset,[banner_box,event,split,video,altitude_min,altitude_max,velocity_min,velocity_max,gimbal_min,gimbal_max,telemetry_panel])
        backend.change(strategy_change,backend,strategy)
        search_button.click(run_search,[query,dataset,backend,strategy,dimension,adaptive,base_dim,top_n,pattern,top_k,repeats,event,split,video,altitude_min,altitude_max,velocity_min,velocity_max,gimbal_min,gimbal_max],[timing,diagnostics,results,download])
    with gr.Tab("Karşılaştır"):
        compare_query=gr.Textbox(value="kalabalık trafik",label="Sorgu"); compare_dataset=gr.Dropdown(DATASETS,value=DATASETS[0] if DATASETS else None,label="Dataset")
        compare_backends=gr.CheckboxGroup(["clickhouse","qdrant","pgvector"],value=["clickhouse","qdrant","pgvector"],label="Backendler")
        compare_dims=gr.CheckboxGroup([2048,1024,512,256],value=[512],label="Boyutlar"); compare_top=gr.Slider(1,50,10,step=1,label="top_k")
        compare_button=gr.Button("Karşılaştır",variant="primary")
        compare_table=gr.Dataframe(headers=["backend","strategy","dimension","p50_ms","p95_ms","recall","returned_count","underfilled","error"])
        compare_button.click(compare,[compare_query,compare_dataset,compare_backends,compare_dims,compare_top],compare_table)


if __name__=="__main__": demo.launch(server_name="0.0.0.0",server_port=7860,show_error=True)
