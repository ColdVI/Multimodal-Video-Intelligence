import csv
import json

from bench.search_run_artifact import (
    SearchRunSpec,
    build_manifest,
    capture_explain_plans,
    write_search_run,
)
from datasets.base import DatasetManifest


def _fake_dataset_manifest():
    return DatasetManifest(
        dataset_id="visdrone", dataset_version="v1", source_hash="abc123",
        split="train", item_count=19, query_count=28, retrieval_unit="video_interval",
        has_structured_filters=True, groundtruth_type="annotation_derived",
        embedding_cache_key="visdrone:abc123",
    )


def _fake_bench_result():
    def row(strategy, mrr, ndcg, mapv, latency, underfilled, agreement=None):
        return {
            "dataset_id": "visdrone", "query_id": f"q-{strategy}", "strategy": strategy,
            "mrr": mrr, "ndcg@10": ndcg, "map": mapv, "n_gt": 3,
            "total_latency_s": latency, "underfilled": underfilled,
            "agreement_vs_2048_exact_at10": agreement,
            "sql": {"single": f"SELECT 1 -- {strategy}"},
        }
    return {
        "dataset_id": "visdrone", "n_queries": 2, "n_sequences": 19,
        "pilot_warning": "Bu sonuç 28 sorguluk pilot değerlendirmedir.",
        "candidate_k_sweep": [20, 50, 100, 200], "final_k": 10,
        "rows": [
            row("2048d_exact", 1.0, 0.9, 0.8, 0.05, False),
            row("2048d_exact", 0.5, 0.4, 0.3, 0.06, False),
            row("adaptive_mrl_256_to_2048", 0.8, 0.7, 0.6, 0.03, True, agreement=0.75),
        ],
    }


def test_run_id_is_deterministic_for_identical_spec():
    spec_a = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    spec_b = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    assert spec_a.run_id == spec_b.run_id


def test_run_id_changes_when_git_sha_changes():
    spec_a = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    spec_b = SearchRunSpec("visdrone", "adaptive_mrl", "cafef00d", (20, 50, 100, 200), 10)
    assert spec_a.run_id != spec_b.run_id


def test_run_id_changes_when_dataset_differs():
    spec_a = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    spec_b = SearchRunSpec("msrvtt_1ka", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    assert spec_a.run_id != spec_b.run_id
    assert "visdrone" in spec_a.run_id
    assert "msrvtt_1ka" in spec_b.run_id


def test_build_manifest_has_all_required_fields():
    spec = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    manifest = build_manifest(spec, _fake_dataset_manifest(), corpus_size=73,
                              model_id="Qwen/Qwen3-VL-Embedding-2B")
    required = {"run_id", "git_sha", "dataset_id", "dataset_hash", "dataset_adapter",
               "retrieval_backend", "model_id", "model_revision", "n_sample", "dimensions",
               "gpu_gate_passed", "embeddings_regenerated", "embeddings_reused_from",
               "corpus_size", "evaluation_power_warning"}
    assert required <= set(manifest)
    assert manifest["corpus_size"] == 73
    assert manifest["embeddings_regenerated"] is False


def test_write_search_run_creates_full_contract_tree(tmp_path):
    spec = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    run_dir = write_search_run(spec, _fake_bench_result(), _fake_dataset_manifest(),
                               corpus_size=73, model_id="Qwen/Qwen3-VL-Embedding-2B",
                               artifacts_root=tmp_path)

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "query_results.ndjson").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
    assert (run_dir / "strategy_matrix.csv").exists()
    assert (run_dir / "sql").is_dir()
    assert (run_dir / "explain").is_dir()

    lines = (run_dir / "query_results.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["strategy"] == "2048d_exact"
    # tam SQL metni (vektor literaliyle) burada TEKRARLANMAZ - sql/ dizininde
    # zaten bir temsili ornek var; satir sadece "sql var mi" bilgisini tasir.
    assert "sql" not in first
    assert first["has_sql"] is True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_power_warning"] == "Bu sonuç 28 sorguluk pilot değerlendirmedir."

    dataset_manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert dataset_manifest["dataset_id"] == "visdrone"
    assert dataset_manifest["query_count"] == 28


def test_metrics_json_aggregates_correctly_by_strategy(tmp_path):
    spec = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    run_dir = write_search_run(spec, _fake_bench_result(), _fake_dataset_manifest(),
                               corpus_size=73, model_id="x", artifacts_root=tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    exact = metrics["by_strategy"]["2048d_exact"]
    assert exact["n_queries"] == 2
    assert exact["underfilled_count"] == 0
    assert exact["mean_mrr"] == 0.75  # (1.0+0.5)/2
    assert exact["mean_agreement_vs_2048_exact_at10"] is None  # referans stratejide agreement yok

    adaptive = metrics["by_strategy"]["adaptive_mrl_256_to_2048"]
    assert adaptive["underfilled_count"] == 1
    assert adaptive["mean_agreement_vs_2048_exact_at10"] == 0.75


def test_strategy_matrix_csv_has_one_row_per_strategy(tmp_path):
    spec = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    run_dir = write_search_run(spec, _fake_bench_result(), _fake_dataset_manifest(),
                               corpus_size=73, model_id="x", artifacts_root=tmp_path)
    with open(run_dir / "strategy_matrix.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    strategies = {r["strategy"] for r in rows}
    assert strategies == {"2048d_exact", "adaptive_mrl_256_to_2048"}


def test_sql_dir_has_one_representative_file_per_strategy(tmp_path):
    spec = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    run_dir = write_search_run(spec, _fake_bench_result(), _fake_dataset_manifest(),
                               corpus_size=73, model_id="x", artifacts_root=tmp_path)
    sql_files = sorted(p.stem for p in (run_dir / "sql").glob("*.json"))
    assert sql_files == ["2048d_exact", "adaptive_mrl_256_to_2048"]


def test_capture_explain_plans_writes_one_file_per_strategy(tmp_path):
    spec = SearchRunSpec("visdrone", "adaptive_mrl", "deadbeef", (20, 50, 100, 200), 10)
    run_dir = write_search_run(spec, _fake_bench_result(), _fake_dataset_manifest(),
                               corpus_size=73, model_id="x", artifacts_root=tmp_path)

    class _FakeExplainResult:
        result_rows = [("Explain plan line",)]

    class _FakeClient:
        def query(self, sql):
            assert sql.startswith("EXPLAIN indexes = 1")
            return _FakeExplainResult()

    capture_explain_plans(run_dir, _FakeClient())
    explain_files = sorted(p.stem for p in (run_dir / "explain").glob("*.json"))
    assert explain_files == ["2048d_exact", "adaptive_mrl_256_to_2048"]
    plan = json.loads((run_dir / "explain" / "2048d_exact.json").read_text(encoding="utf-8"))
    assert plan["single"] == "Explain plan line"
