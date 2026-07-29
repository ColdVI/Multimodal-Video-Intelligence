"""MSR-VTT 1k-A icin somut DatasetAdapter. scripts/validate_msrvtt.py'nin
BILEREK ClickHouse'a yazmayan protokolunu (tek pencere = tum klip, T2V
retrieval) sarar - bkz. o dosyanin docstring'i. Bu adapter ClickHouse'a
YAZMAZ; retrieval_backend='artifact_matrix' (config.yaml: datasets.
msrvtt_1ka) bunu acikca isaretler."""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

from .base import DatasetAdapter, DatasetManifest


class MSRVTTAdapter(DatasetAdapter):
    dataset_id = "msrvtt_1ka"
    name = "MSR-VTT 1k-A test split"
    has_captions = True

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._entries_cache = None

    def _split_path(self) -> pathlib.Path:
        return pathlib.Path(self.cfg["datasets"]["msrvtt_1ka"]["split"])

    def _entries(self) -> list:
        if self._entries_cache is None:
            self._entries_cache = json.load(open(self._split_path(), encoding="utf-8"))
        return self._entries_cache

    def _by_video_id(self) -> dict:
        return {e["video_id"]: e for e in self._entries()}

    def list_sequences(self) -> list:
        return [e["video_id"] for e in self._entries()]

    def load_video(self, seq_id: str) -> pathlib.Path:
        entry = self._by_video_id()[seq_id]
        return pathlib.Path(self.cfg["datasets"]["msrvtt_1ka"]["videos_dir"]) / entry["video"]

    def fps(self, seq_id: str) -> float:
        """MSR-VTT'de VisDrone gibi onceden hesaplanmis bir fps manifesti
        yok - video dosyasindan gercek-zamanli okunur (bkz.
        scripts/validate_msrvtt.py::probe_video_duration ile ayni desen)."""
        import cv2
        cap = cv2.VideoCapture(str(self.load_video(seq_id)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps

    def ground_truth(self, seq_id: str) -> dict:
        """DIKKAT: VisDrone'daki 'query -> alt-interval' anlaminda DEGIL -
        MSR-VTT'de retrieval birimi TUM KLIP (retrieval_unit='video'),
        alt-interval lokalizasyonu yok. Burada {caption: [(0.0, sure)]}
        donuyoruz ki arayuz sozlesmesi (query -> zaman araliklari) bozulmasin,
        ama bu 've video'nun tamami dogru cevap' anlamina gelir, saniye-
        hassasiyetinde bir zaman araligi anlamina GELMEZ - eval raporu bu
        ayrimi acikca belirtmeli (bkz. DatasetAdapter.ground_truth
        docstring'i, protokoller birbirinden farkli)."""
        entry = self._by_video_id()[seq_id]
        duration = max(0.0, entry["end time"] - entry["start time"])
        return {entry["caption"]: [(0.0, duration)]}

    def captions(self, seq_id: str):
        return [self._by_video_id()[seq_id]["caption"]]

    def license(self) -> str:
        return self.cfg["datasets"]["msrvtt_1ka"]["license"]

    def manifest(self) -> DatasetManifest:
        entries = self._entries()
        split_hash = hashlib.sha256(self._split_path().read_bytes()).hexdigest()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_version=self.cfg["datasets"]["msrvtt_1ka"]["dataset_version"],
            source_hash=split_hash,
            split="1k-A test",
            item_count=len(entries),
            query_count=len(entries),  # video basina tek caption (1k-A protokolu)
            retrieval_unit="video",
            has_structured_filters=False,
            groundtruth_type="caption_1to1",
            embedding_cache_key=f"msrvtt_1ka:{split_hash[:12]}",
        )


__all__ = ["MSRVTTAdapter"]
