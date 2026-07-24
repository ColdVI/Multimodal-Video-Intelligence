"""Iki eksen: model x filtre(acik/kapali). Her (model,filtre,sorgu) icin
hem ozet metrik (results.json) hem FiftyOne icin detay
(results_detail.json) yazar."""
import argparse
import itertools
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config
from eval.metrics import evaluate
from models import available_models
from search.merge import merge_intervals
from search.query import search

MODELS = ["siglip2_frameavg", "xclip_hf_zeroshot"]  # adapter eklendikce buyur


def category_of(q: str) -> str:
    if " ve " in q or "birlikte" in q:
        return "bileşik"
    if "yürüyen" in q or "koşan" in q:
        return "hareket"
    return "tekli"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", action="append", dest="models", choices=available_models(),
        help="yalnizca bu modeli degerlendir; birden cok kez verilebilir",
    )
    args = ap.parse_args()

    cfg = load_config()
    gt_path = pathlib.Path(cfg["paths"]["groundtruth"])
    if not gt_path.exists():
        print(f"HATA: {gt_path} yok. Once eval/make_groundtruth.py calistirin.")
        raise SystemExit(1)
    gt_all = json.load(open(gt_path))

    k = cfg["eval"]["top_k"]
    iou_thr = cfg["eval"]["iou_threshold"]
    gap_tol = cfg["merge"]["gap_tolerance_s"]

    summary_rows = []
    detail_rows = []

    models = args.models or MODELS
    for model, use_filter in itertools.product(models, [True, False]):
        for q, gt_by_vid in gt_all.items():
            raw = search(q, model, use_filters=use_filter)
            pred = merge_intervals(raw, gap_tol=gap_tol)
            metrics = evaluate(pred, gt_by_vid, k=k, iou_thr=iou_thr)
            summary_rows.append({
                "model": model, "filter": use_filter, "query": q,
                "category": category_of(q), **metrics,
            })

            pred_by_vid = defaultdict(list)
            for vid, t0, t1, s in pred:
                pred_by_vid[vid].append((t0, t1, s))
            for vid in set(pred_by_vid) | set(gt_by_vid):
                detail_rows.append({
                    "model": model, "filter": use_filter, "query": q,
                    "video_id": vid,
                    "pred": pred_by_vid.get(vid, []),
                    "gt": gt_by_vid.get(vid, []),
                })

    json.dump(summary_rows, open("results.json", "w"), indent=1)
    json.dump(detail_rows, open("results_detail.json", "w"), indent=1)
    print(f"{len(summary_rows)} ozet satir -> results.json")
    print(f"{len(detail_rows)} detay satir -> results_detail.json")


if __name__ == "__main__":
    main()
