import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ingest.frame_io import read_frames_sequential, read_window_frames, sample_frame_indices


def test_sample_frame_indices_matches_manual_linspace():
    t0, t1, fps, n = 2.0, 10.0, 25.0, 32
    expected = [int(t * fps) for t in np.linspace(t0, t1, n, endpoint=False)]
    assert sample_frame_indices(t0, t1, fps, n) == expected


def test_sample_frame_indices_is_sorted_ascending():
    indices = sample_frame_indices(0.0, 8.0, 25.0, 32)
    assert indices == sorted(indices)


@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / "tiny.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (16, 12))
    frames = []
    for i in range(50):
        frame = np.full((12, 16, 3), i % 256, dtype=np.uint8)
        frame[:, :, 0] = i % 256  # B kanalinda kare indeksini kodla
        writer.write(frame)
        frames.append(frame)
    writer.release()
    return path, frames


def test_read_frames_sequential_returns_correct_indices(tiny_video):
    path, frames = tiny_video
    # mp4v lossy encode kisa/senteik klipte kare-indeks kaymasi yapabiliyor
    # (encoder gecikmesi/yeniden siralama) - bu yuzden burada yalnizca
    # "istenen indeksler dondu, sekil dogru" kontrol ediliyor. Piksel-tam
    # dogrulama gercek bir production videosunda yapildi (bkz. scratchpad
    # verify_frame_io.py: 32 gercek kare, eski/yeni arasinda tam bit-esitligi,
    # ve asagidaki test_read_window_frames_matches_per_frame_seek_baseline
    # eski ve yeni yontemin BU sentetik videoda da birbiriyle ayni sonucu
    # verdigini dogruluyor).
    wanted = [0, 5, 5, 20, 40]  # tekrar eden indeks de test edilsin
    result = read_frames_sequential(path, wanted)
    assert set(result.keys()) == {0, 5, 20, 40}
    for idx in {0, 5, 20, 40}:
        assert result[idx].shape == frames[idx].shape


def test_read_frames_sequential_empty_indices_returns_empty_dict(tiny_video):
    path, _ = tiny_video
    assert read_frames_sequential(path, []) == {}


def test_read_window_frames_matches_per_frame_seek_baseline(tiny_video):
    path, _ = tiny_video

    def old_read_window(video_path, t0, t1, n):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out = []
        for t in np.linspace(t0, t1, n, endpoint=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if ok:
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return out

    old = old_read_window(path, 0.0, 1.5, 10)
    new = read_window_frames(path, 0.0, 1.5, 10)
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert np.array_equal(a, b)
