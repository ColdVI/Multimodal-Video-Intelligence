import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "notebooks" / "inspect_fiftyone.py"
SPEC = importlib.util.spec_from_file_location("inspect_fiftyone", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_interval_to_support_is_one_indexed_and_clamped():
    assert MODULE.interval_to_support(0.0, 8.0, 25, 213) == [1, 200]
    assert MODULE.interval_to_support(4.0, 8.52, 25, 213) == [101, 213]
    assert MODULE.interval_to_support(0.0, 8.76, 25, 219) == [1, 219]
