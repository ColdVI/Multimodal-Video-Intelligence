"""Paylasilan video kare okuma - ingest/03_embed.py, ingest/04_detect.py ve
scripts/colab_gpu_bench.py'nin ucu de ayni "linspace(t0,t1,n) -> int(t*fps)"
orneklemesini ve cv2 okumasini tekrar ediyordu.

NEDEN: cv2.CAP_PROP_POS_FRAMES ile HER kareyi ayri ayri "seek" etmek
(eski desen), H.264'te decoder'i en yakin onceki keyframe'e geri gonderip
oradan ileri decode etmeye zorluyor. Bir pencerede n_sample kare istendiginde
bu n_sample kez tekrarlaniyordu - GPU benchmarkinda pencere basina 7-205s
olculdu (bkz. docs/codex/06_NIHAI_RAPOR.md). Duzeltme: pencerenin ilk
gerekli karesine TEK seek, sonrasi sequential grab/read - toplam decode
isi ayni ama keyframe-geri-donus sayisi n_sample'dan 1'e dusuyor.
"""
import cv2
import numpy as np


def sample_frame_indices(t0: float, t1: float, fps: float, n: int) -> list:
    """np.linspace(t0, t1, n, endpoint=False) -> int(t*fps), sirali (artan)
    liste. Eskiden ingest/03_embed.py, ingest/04_detect.py ve
    scripts/colab_gpu_bench.py icinde ayri ayri inline yaziliydi."""
    return [int(t * fps) for t in np.linspace(t0, t1, n, endpoint=False)]


def read_frames_sequential(video_path: str, frame_indices) -> dict:
    """Verilen kare indekslerini TEK seek + sequential grab/read ile okur.
    Donen: {frame_index: RGB ndarray}. Bulunamayan/okunamayan indeksler
    sozlukte yer almaz (eski per-frame-seek davranisiyla ayni - `ok` False
    donerse o kare atlanirdi).

    frame_indices tekrar icerebilir (ayni indeks birden fazla istenmis
    olabilir) - sozluk dogal olarak dedupe eder, cagiran taraf kendi
    sirasina gore sozlukten tekrar okur."""
    unique_sorted = sorted(set(int(i) for i in frame_indices))
    if not unique_sorted:
        return {}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, unique_sorted[0])
    result = {}
    pos = unique_sorted[0]
    for target in unique_sorted:
        while pos < target:
            if not cap.grab():
                cap.release()
                return result
            pos += 1
        ok, frame = cap.read()
        if not ok:
            break
        result[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pos += 1
    cap.release()
    return result


def read_window_frames(video_path: str, t0: float, t1: float, n: int) -> list:
    """read_window()/window_features() icin dogrudan degistirme: eski
    "her kare icin ayri seek" dongusunun yerine gecer, ayni sirali frame
    listesini dondurur (bulunamayan kareler atlanir, eskisi gibi)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    indices = sample_frame_indices(t0, t1, fps, n)
    frame_map = read_frames_sequential(video_path, indices)
    return [frame_map[i] for i in indices if i in frame_map]
