import json

from bench.report import aggregate_by_category, write_report


def _fake_row(query, category, recall=1.0, precision=1.0, mrr=1.0, n_gt=1):
    by_k = {k: {"recall@k": recall, "precision@k": precision, "n_gt": n_gt,
                "n_pred": 1, "n_hits": 1} for k in (1, 5, 10)}
    return {"query": query, "category": category, "lang": "tr", "concept": "bus",
            "n_pred_windows": 1, "n_pred_intervals": 1, "mrr": mrr, "n_gt": n_gt,
            "by_k": by_k}


def _fake_run(run_id="m_filt_auto_cpu_abc123"):
    rows = [_fake_row("otobüsü göster", "tekli"), _fake_row("show the bus", "tekli", recall=0.5)]
    return {
        "run_id": run_id,
        "result": {
            "spec": {"model_name": "xclip_hf_zeroshot", "use_filters": True,
                     "strategy": "auto", "top_k": 200, "hardware_profile": "cpu",
                     "yolo_variant": "yolo26x"},
            "rows": rows,
            "timing": {"query": {"n": 2, "total_s": 0.1, "mean_s": 0.05,
                                  "p50_s": 0.05, "p95_s": 0.05}},
        },
        "manifest": {"duration_s": 1.23, "run_id": run_id},
    }


def test_aggregate_by_category_averages_within_category():
    rows = [_fake_row("q1", "tekli", recall=1.0), _fake_row("q2", "tekli", recall=0.5)]
    agg = aggregate_by_category(rows)
    assert agg["tekli"]["recall@10"] == 0.75


def test_aggregate_by_category_separates_categories():
    rows = [_fake_row("q1", "tekli", recall=1.0), _fake_row("q2", "hareket", recall=0.0)]
    agg = aggregate_by_category(rows)
    assert set(agg) == {"tekli", "hareket"}


def test_write_report_produces_valid_json_and_html(tmp_path):
    run = _fake_run()
    html_path, json_path = write_report([run], out_dir=str(tmp_path))

    payload = json.loads(open(json_path, encoding="utf-8").read())
    assert payload[0]["run_id"] == run["run_id"]
    assert payload[0]["category_summary"]["tekli"]["recall@10"] == 0.75

    html = open(html_path, encoding="utf-8").read()
    assert "xclip_hf_zeroshot" in html
    assert run["run_id"] in html
