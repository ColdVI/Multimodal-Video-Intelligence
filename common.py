"""Paylaşılan config yükleyici. Tüm sayısal sabitlerin tek kaynağı config.yaml
- kodun içinde hardcoded pencere/eşik değeri olmamalı."""
import functools
import os
import pathlib

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parent


def configure_runtime_environment(runtime_dir=None):
    base = pathlib.Path(
        runtime_dir
        or os.environ.get('VIDEO_SEARCH_RUNTIME_DIR', REPO_ROOT / '.runtime')
    ).resolve()
    base.mkdir(parents=True, exist_ok=True)
    defaults = {
        'YOLO_CONFIG_DIR': base,
        'HF_HOME': base / 'huggingface',
        'FIFTYONE_CONFIG_PATH': base / 'fiftyone' / 'config.json',
        'FIFTYONE_DATABASE_DIR': base / 'fiftyone' / 'db',
        'FIFTYONE_DEFAULT_DATASET_DIR': base / 'fiftyone' / 'datasets',
        'FIFTYONE_DATASET_ZOO_DIR': base / 'fiftyone' / 'zoo' / 'datasets',
        'FIFTYONE_MODEL_ZOO_DIR': base / 'fiftyone' / 'zoo' / 'models',
        'FIFTYONE_PLUGINS_DIR': base / 'fiftyone' / 'plugins',
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    return base


@functools.lru_cache(maxsize=1)
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
