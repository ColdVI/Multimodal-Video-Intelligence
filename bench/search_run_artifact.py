"""artifacts/search_runs/<run_id>/ sozlesmesini yazar (unified_search_
harness_duzeltmeler.md #7). Canli hesap yapmaz - run_adaptive_mrl_bench()
gibi bir harness'in ciktisini (rows + meta) alir, dosyaya doker."""
import csv
import dataclasses
import hashlib
import json
import pathlib
import statistics

from bench.manifest import git_hash

_NUMERIC_METRIC_KEYS = ("mrr", "ndcg@10", "map", "total_latency_s")


@dataclasses.dataclass(frozen=True)
class SearchRunSpec:
    """run_id, mevcut bench/spec.py::RunSpec ile AYNI desen: dataclass
    alanlarinin SHA1'i. Ayni girdi (ayni git commit + ayni dataset + ayni
    sweep) HER ZAMAN ayni run_id'yi uretir (bkz. tests/test_search_run_
    artifact.py::test_run_id_is_deterministic) - farkli git_sha (kod
    degisti) FARKLI run_id demektir, eski run'in ustune sessizce yazilmaz."""
    dataset_id: str
    harness: str
    git_sha: str
    candidate_k_sweep: tuple
    final_k: int

    @property
    def run_id(self) -> str:
        payload = "|".join(str(v) for v in dataclasses.astuple(self))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        return f"{self.harness}_{self.dataset_id}_{digest}"


def _aggregate_by_strategy(rows: list) -> dict:
    by_strategy = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(r)
    out = {}
    for strategy, strategy_rows in by_strategy.items():
        agreements = [r["agreement_vs_2048_exact_at10"] for r in strategy_rows
                     if r["agreement_vs_2048_exact_at10"] is not None]
        out[strategy] = {
            "n_queries": len(strategy_rows),
            "underfilled_count": sum(1 for r in strategy_rows if r["underfilled"]),
            "mean_mrr": statistics.fmean(r["mrr"] for r in strategy_rows),
            "mean_ndcg@10": statistics.fmean(r["ndcg@10"] for r in strategy_rows),
            "mean_map": statistics.fmean(r["map"] for r in strategy_rows),
            "mean_total_latency_s": statistics.fmean(r["total_latency_s"] for r in strategy_rows),
            "mean_agreement_vs_2048_exact_at10": (
                statistics.fmean(agreements) if agreements else None),
        }
    return out


def build_manifest(spec: SearchRunSpec, dataset_manifest, corpus_size: int,
                   model_id: str, extra: dict = None) -> dict:
    manifest = {
        "run_id": spec.run_id,
        "git_sha": spec.git_sha,
        "dataset_id": spec.dataset_id,
        "dataset_hash": dataset_manifest.source_hash,
        "dataset_adapter": dataset_manifest.dataset_id,
        "retrieval_backend": "clickhouse",
        "model_id": model_id,
        "model_revision": "unknown",  # sentence-transformers checkpoint revision pin'lenmedi
        "n_sample": 6,  # qwen3vl_emb.embed_video varsayilani - bu kosumda YENIDEN uretilmedi, belgeleme amacli
        "dimensions": [256, 512, 1024, 2048],
        "gpu_gate_passed": None,  # bu kosum GPU-gated bir video-embed adimi icermiyor
        "embeddings_regenerated": False,
        "embeddings_reused_from": "data/embeddings_qwen3vl_emb_2048.json (+ mrl_truncate_embeddings.py turevleri)",
        "corpus_size": corpus_size,
        "evaluation_power_warning": None,  # caller (write_search_run) doldurur
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_search_run(spec: SearchRunSpec, bench_result: dict, dataset_manifest,
                     corpus_size: int, model_id: str,
                     artifacts_root: pathlib.Path = pathlib.Path("artifacts/search_runs")) -> pathlib.Path:
    run_dir = artifacts_root / spec.run_id
    (run_dir / "sql").mkdir(parents=True, exist_ok=True)
    (run_dir / "explain").mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(spec, dataset_manifest, corpus_size, model_id,
                              extra={"evaluation_power_warning": bench_result.get("pilot_warning")})
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Tam SQL metni (2048d vektor literali dahil, satir basina ~20KB) burada
    # TEKRARLANMAZ - sql/<strategy>.json zaten bir temsili ornek tutuyor.
    # Satirda sadece SQL var/yok bilgisi (has_sql) kalir.
    with open(run_dir / "query_results.ndjson", "w", encoding="utf-8") as f:
        for row in bench_result["rows"]:
            slim_row = {k: v for k, v in row.items() if k != "sql"}
            slim_row["has_sql"] = bool(row.get("sql"))
            f.write(json.dumps(slim_row, ensure_ascii=False, default=str) + "\n")

    by_strategy = _aggregate_by_strategy(bench_result["rows"])
    (run_dir / "metrics.json").write_text(
        json.dumps({"by_strategy": by_strategy, "n_queries": bench_result["n_queries"],
                   "n_rows": len(bench_result["rows"])}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(dataclasses.asdict(dataset_manifest), indent=2, ensure_ascii=False),
        encoding="utf-8")

    with open(run_dir / "strategy_matrix.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["strategy", "n_queries", "underfilled_count", "mean_mrr", "mean_ndcg@10",
               "mean_map", "mean_total_latency_s", "mean_agreement_vs_2048_exact_at10"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for strategy, agg in by_strategy.items():
            writer.writerow({"strategy": strategy, **agg})

    # SQL/EXPLAIN: her stratejiden BIR temsili ornek (392 satirin tumu degil) -
    # sadece ClickHouse rejimindeki alanlar (retrieval_backend='clickhouse').
    seen_strategies = set()
    for row in bench_result["rows"]:
        if row["strategy"] in seen_strategies:
            continue
        seen_strategies.add(row["strategy"])
        sql = row.get("sql")
        if sql:
            (run_dir / "sql" / f"{row['strategy']}.json").write_text(
                json.dumps(sql, indent=2, ensure_ascii=False), encoding="utf-8")

    return run_dir


def capture_explain_plans(run_dir: pathlib.Path, client) -> None:
    """run_dir/sql/*.json'daki (zaten yazilmis, temsili) sorgular icin
    EXPLAIN indexes=1 yakalar ve run_dir/explain/'e yazar - AYRI adim,
    cunku write_search_run() canli client GEREKTIRMEMELI (testte fake
    bench_result ile client'siz cagrilabilsin diye)."""
    sql_dir = run_dir / "sql"
    if not sql_dir.exists():
        return
    for sql_file in sql_dir.glob("*.json"):
        strategy = sql_file.stem
        sql_map = json.loads(sql_file.read_text(encoding="utf-8"))
        plans = {}
        for stage, sql in sql_map.items():
            if not sql:
                continue
            try:
                rows = client.query("EXPLAIN indexes = 1\n" + sql.rstrip()).result_rows
                plans[stage] = "\n".join(str(r[0]) for r in rows)
            except Exception as exc:  # ClickHouse surumune gore EXPLAIN sozdizimi degisebilir
                plans[stage] = f"EXPLAIN basarisiz: {exc}"
        (run_dir / "explain" / f"{strategy}.json").write_text(
            json.dumps(plans, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["SearchRunSpec", "build_manifest", "write_search_run", "capture_explain_plans"]
