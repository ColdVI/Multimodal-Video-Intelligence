import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "download_visdrone.py"
SPEC = importlib.util.spec_from_file_location("download_visdrone", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeInfo:
    def __init__(self, filename):
        self.filename = filename


class FakeArchive:
    def __init__(self, names):
        self.infos = [FakeInfo(name) for name in names]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def infolist(self):
        return self.infos


def test_validate_archive_accepts_required_contract():
    archive = FakeArchive([
        "VisDrone2019-MOT-train/sequences/a/0000001.jpg",
        "VisDrone2019-MOT-train/annotations/a.txt",
    ])

    with patch.object(MODULE.zipfile, "ZipFile", return_value=archive):
        assert MODULE.validate_archive(
            Path("sample.zip"), expected_bytes=None, expected_sha256=None
        ) == 2


def test_validate_archive_rejects_path_traversal():
    archive = FakeArchive([
        "VisDrone2019-MOT-train/sequences/a/x.jpg",
        "VisDrone2019-MOT-train/annotations/a.txt",
        "../escape.txt",
    ])

    with patch.object(MODULE.zipfile, "ZipFile", return_value=archive):
        with pytest.raises(RuntimeError, match="guvensiz ZIP yolu"):
            MODULE.validate_archive(
                Path("unsafe.zip"), expected_bytes=None, expected_sha256=None
            )
