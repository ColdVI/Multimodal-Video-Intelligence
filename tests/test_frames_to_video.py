import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ingest" / "01_frames_to_video.py"
SPEC = importlib.util.spec_from_file_location("frames_to_video", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_sequences_filters_and_rejects_unknown():
    class FakePath:
        def __init__(self, name, is_dir=True):
            self.name = name
            self._is_dir = is_dir

        def is_dir(self):
            return self._is_dir

        def __lt__(self, other):
            return self.name < other.name

    class Raw:
        def iterdir(self):
            return [FakePath("b"), FakePath("a"), FakePath("file", False)]

    assert [p.name for p in MODULE.select_sequences(Raw())] == ["a", "b"]
    assert [p.name for p in MODULE.select_sequences(Raw(), ["b"])] == ["b"]
    with pytest.raises(ValueError, match="missing"):
        MODULE.select_sequences(Raw(), ["missing"])


def test_ffmpeg_command_pads_odd_dimensions_for_yuv420p():
    command = MODULE.build_ffmpeg_command(
        "ffmpeg", Path("sequence"), Path("video.mp4")
    )
    assert command[command.index("-vf") + 1] == "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
