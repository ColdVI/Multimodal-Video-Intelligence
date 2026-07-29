"""Dataset adapter registry. models/__init__.py::get_embedder ile ayni
desen: get_dataset_adapter(dataset_id) yoksa olusturur, varsa onbellekten
doner. dataset_id (config.yaml: datasets.<id>) ile adapter sinifi
(config.yaml: datasets.<id>.adapter) AYRI kavramlar - bir adapter sinifi
teorik olarak birden fazla dataset_id altinda (farkli split/versiyon)
kullanilabilir."""
from .base import DatasetAdapter, DatasetManifest, qualified_id
from .msrvtt import MSRVTTAdapter
from .registry import (
    dataset_config,
    dataset_configs,
    list_dataset_ids,
    run_strategy_or_unsupported,
    supports_strategy,
)
from .visdrone import VisDroneAdapter

_ADAPTER_CLASSES = {
    "visdrone": VisDroneAdapter,
    "msrvtt": MSRVTTAdapter,
}
_instances = {}


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    if dataset_id not in _instances:
        cfg = dataset_config(dataset_id)
        adapter_name = cfg["adapter"]
        if adapter_name not in _ADAPTER_CLASSES:
            raise KeyError(
                f"Bilinmeyen adapter '{adapter_name}' (dataset_id={dataset_id!r}). "
                f"Mevcut: {sorted(_ADAPTER_CLASSES)}"
            )
        _instances[dataset_id] = _ADAPTER_CLASSES[adapter_name]()
    return _instances[dataset_id]


def available_datasets() -> list:
    return list_dataset_ids()


__all__ = [
    "DatasetAdapter", "DatasetManifest", "qualified_id",
    "dataset_config", "dataset_configs", "list_dataset_ids",
    "supports_strategy", "run_strategy_or_unsupported",
    "get_dataset_adapter", "available_datasets",
]
