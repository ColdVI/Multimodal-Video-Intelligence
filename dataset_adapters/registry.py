"""Config-driven dataset registry (config.yaml: datasets:). Amac: yeni bir
dataset eklemek kod dallanmasi (`if dataset == "visdrone": ...`) degil,
config.yaml'a bir girdi + datasets/<ad>.py adaptoru eklemek olsun - bkz.
unified_search_harness_duzeltmeler.md #2.1."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

_REQUIRED_FIELDS = {
    "adapter", "retrieval_backend", "has_structured_filters",
    "has_temporal_windows", "evaluation_regime", "query_count",
}
# retrieval_backend='artifact_matrix' olan dataset'lerde ClickHouse'a ozgu
# stratejiler (filtreye dayali prefilter/postfilter, telemetri filtresi)
# YAPISAL olarak anlamsiz - has_structured_filters=False garantisiyle tutarli.
_STRUCTURED_FILTER_STRATEGIES = {"prefilter", "postfilter_rescore", "telemetry_filter"}


def dataset_configs() -> dict:
    """config.yaml: datasets: bolumunu okur, her girdiyi zorunlu alanlar
    icin dogrular. Eksik alan varsa acikca patlar - sessiz varsayilan yok."""
    cfg = load_config()
    datasets = cfg.get("datasets", {})
    for dataset_id, entry in datasets.items():
        missing = _REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"datasets.{dataset_id}: eksik zorunlu alan(lar): {sorted(missing)}")
    return datasets


def dataset_config(dataset_id: str) -> dict:
    configs = dataset_configs()
    if dataset_id not in configs:
        raise KeyError(f"Bilinmeyen dataset_id {dataset_id!r}. Mevcut: {sorted(configs)}")
    return configs[dataset_id]


def list_dataset_ids() -> list:
    return sorted(dataset_configs())


def supports_strategy(dataset_id: str, strategy: str) -> bool:
    """Bu dataset'in retrieval backend'i verilen stratejiyi destekliyor mu.
    'clickhouse' backend tum stratejileri destekler (mevcut VisDrone
    davranisi). 'artifact_matrix' backend (ör. MSR-VTT) yapisal filtre
    kolonlarina sahip olmadigi icin filtre-bagimli stratejileri desteklemez -
    caller bunu sessizce yoksaymak yerine acik bir 'unsupported_strategy'
    sonucu uretmeli (bkz. run_strategy_or_unsupported())."""
    cfg = dataset_config(dataset_id)
    if cfg["retrieval_backend"] == "clickhouse":
        return True
    return strategy not in _STRUCTURED_FILTER_STRATEGIES


def run_strategy_or_unsupported(dataset_id: str, strategy: str, run_fn):
    """supports_strategy() Falseysa run_fn'i HIC CAGIRMADAN acik bir
    unsupported_strategy sonucu doner - sessiz fallback/no-op YASAK
    (unified_search_harness_duzeltmeler.md #2.3)."""
    if not supports_strategy(dataset_id, strategy):
        return {
            "unsupported_strategy": True,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "reason": (f"{dataset_id} datasetinin retrieval_backend'i "
                      f"({dataset_config(dataset_id)['retrieval_backend']}) "
                      f"'{strategy}' stratejisini desteklemiyor - yapisal filtre "
                      "kolonu yok (has_structured_filters=False)."),
        }
    return run_fn()


__all__ = ["dataset_configs", "dataset_config", "list_dataset_ids",
          "supports_strategy", "run_strategy_or_unsupported"]
