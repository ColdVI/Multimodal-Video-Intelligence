"""RunSpec -> pipeline'in ilgili kismini kosar -> artifacts/bench/<run_id>/
altina yazar (Faz 1 madde 1). Bu faz query+eval asamasini kapsar; ingest
(frames/windows/detect/embed/load) ayri, `ingest` Makefile hedefiyle
onceden hazirlanmis veriye dayanir - "Bu faz bitmeden model/DB deneyi yok"
kuralindan once altyapinin kendisi dogrulanir."""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import load_config
from eval.make_groundtruth import build_queries, build_query_metadata, load_annotations
from search.merge import merge_intervals
from search.query import search

from bench.manifest import capture_run_manifest
from bench.metrics import evaluate_multi_k
from bench.spec import RunSpec
from bench.timing import StageTimer


def compute_gt(cfg: dict, sequence_ids=None):
    """eval/make_groundtruth.py::main()'in bellek-ici versiyonu - dosyaya
    yazmadan, istenirse bir sekans alt kumesine sinirlanarak GT uretir.
    Ayni annotation kaynagi, ayni GT fonksiyonlari - tek sozlesme."""
    manifest = json.load(open(cfg["paths"]["manifest"]))
    if sequence_ids is not None:
        manifest = {k: v for k, v in manifest.items() if k in sequence_ids}
    ann_dir = pathlib.Path(cfg["paths"]["annotations_dir"])
    queries = build_queries()

    gt = {q: {} for q in queries}
    for vid, m in manifest.items():
        ann_path = ann_dir / f"{vid}.txt"
        if not ann_path.exists():
            continue
        frames = load_annotations(ann_path)
        for q, fn in queries.items():
            iv = fn(frames, m["n_frames"], m["fps"])
            if iv:
                gt[q][vid] = iv
    return gt, sorted(manifest)


def run_one(spec: RunSpec, cfg: dict, gt: dict, query_meta: dict,
            timer: StageTimer = None) -> dict:
    timer = timer if timer is not None else StageTimer()
    rows = []
    for q, gt_by_vid in gt.items():
        with timer.measure("query"):
            raw = search(q, spec.model_name, top_k=spec.top_k,
                         use_filters=spec.use_filters)
        with timer.measure("merge"):
            pred = merge_intervals(raw, gap_tol=cfg["merge"]["gap_tolerance_s"])
        metrics = evaluate_multi_k(pred, gt_by_vid)
        meta = query_meta.get(q, {})
        rows.append({
            "query": q,
            "category": meta.get("category"),
            "lang": meta.get("lang"),
            "concept": meta.get("concept"),
            "n_pred_windows": len(raw),
            "n_pred_intervals": len(pred),
            **metrics,
        })
    return {"spec": spec.as_dict(), "rows": rows, "timing": timer.summary()}


def run_determinism_check(spec: RunSpec, cfg: dict, gt: dict, query_meta: dict,
                          tol: float = 1e-6) -> dict:
    """Ayni RunSpec'i iki kez kosar; her sorgunun by_k metriklerinin ve
    MRR'inin (tol icinde) ayni ciktigini dogrular - Faz 1 kanit gereksinimi."""
    first = run_one(spec, cfg, gt, query_meta)
    second = run_one(spec, cfg, gt, query_meta)
    mismatches = []
    for r1, r2 in zip(first["rows"], second["rows"]):
        if r1["by_k"] != r2["by_k"] or abs(r1["mrr"] - r2["mrr"]) > tol:
            mismatches.append(r1["query"])
    return {"deterministic": not mismatches, "mismatched_queries": mismatches,
            "n_queries": len(first["rows"])}


def run_all(specs: list, subset_sequences: list = None,
           artifacts_dir: str = "artifacts/bench") -> list:
    cfg = load_config()
    gt, sequence_ids = compute_gt(cfg, subset_sequences)
    query_meta = build_query_metadata()
    out_root = pathlib.Path(artifacts_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in specs:
        timer = StageTimer()
        t0 = time.perf_counter()
        result = run_one(spec, cfg, gt, query_meta, timer=timer)
        duration_s = time.perf_counter() - t0
        manifest = capture_run_manifest(
            spec, cfg,
            data_scope={"sequences": sequence_ids, "n_sequences": len(sequence_ids),
                       "n_queries": len(gt)},
            duration_s=duration_s,
        )
        run_dir = out_root / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        results.append({"run_id": spec.run_id, "result": result, "manifest": manifest})
        print(f"run tamam: {spec.run_id} ({duration_s:.1f} sn, {len(gt)} sorgu)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-determinism", action="store_true",
                    help="Rapor uretmek yerine bir RunSpec'i iki kez kosup "
                         "sonuclarin ayni ciktigini dogrular (Faz 1 kanit gereksinimi).")
    args = ap.parse_args()

    cfg = load_config()
    bench_subset = cfg.get("bench", {}).get("subset")

    if args.check_determinism:
        gt, sequence_ids = compute_gt(cfg, bench_subset)
        query_meta = build_query_metadata()
        spec = RunSpec(model_name="xclip_hf_zeroshot", use_filters=True)
        outcome = run_determinism_check(spec, cfg, gt, query_meta)
        print(f"determinizm kontrolu ({spec.run_id}, {len(sequence_ids)} sekans, "
             f"{outcome['n_queries']} sorgu): "
             f"{'GECTI' if outcome['deterministic'] else 'BASARISIZ'}")
        if not outcome["deterministic"]:
            print("uyusmayan sorgular:", outcome["mismatched_queries"])
            raise SystemExit(1)
        return

    models = ["xclip_hf_zeroshot", "siglip2_frameavg"]
    specs = [RunSpec(model_name=m, use_filters=f)
            for m in models for f in (True, False)]
    results = run_all(specs, subset_sequences=bench_subset)

    from bench.report import write_report
    html_path, json_path = write_report(results)
    print(f"HTML rapor: {html_path}")
    print(f"JSON kanit: {json_path}")


if __name__ == "__main__":
    main()
