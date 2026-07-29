"""VisDrone2019-MOT icin somut DatasetAdapter. ClickHouse ingest hattini
(ingest/*.py) DEGISTIRMEZ - bench/eval/artifact gibi dataset-agnostik
koda ortak bir okuma yuzeyi saglar. GT, eval/make_groundtruth.py'nin
CANLI build_queries()'inden turer - data/groundtruth/gt.json (eski, 6
kayitlik bir smoke-donemi ciktisi) hic okunmaz."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config
from eval.make_groundtruth import build_queries, load_annotations

from .base import DatasetAdapter, DatasetManifest

# STATUS.md'de belgelenmis, gercek indirme dogrulamasindan (bu oturumda
# yeniden hesaplanmadi - kaynak: "Resmi Task 4 VisDrone-MOT trainset Google
# Drive'dan indirildi: 8.080.572.990 bayt, SHA-256 ...").
_SOURCE_ZIP_SHA256 = "566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014"


class VisDroneAdapter(DatasetAdapter):
    dataset_id = "visdrone"
    name = "VisDrone2019-MOT-train"
    has_captions = False

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._manifest_cache = None
        self._queries_cache = None

    def _manifest_data(self) -> dict:
        if self._manifest_cache is None:
            self._manifest_cache = json.load(
                open(self.cfg["paths"]["manifest"], encoding="utf-8"))
        return self._manifest_cache

    def _queries(self) -> dict:
        if self._queries_cache is None:
            self._queries_cache = build_queries()
        return self._queries_cache

    def list_sequences(self) -> list:
        return sorted(self._manifest_data())

    def load_video(self, seq_id: str) -> pathlib.Path:
        return pathlib.Path(self.cfg["paths"]["videos_dir"]) / f"{seq_id}.mp4"

    def fps(self, seq_id: str) -> float:
        return self._manifest_data()[seq_id]["fps"]

    def ground_truth(self, seq_id: str) -> dict:
        """query -> [(t0,t1),...], annotation'dan turetilmis (bkz.
        eval/make_groundtruth.py). Sadece bu seq_id icin hesaplanir - GT
        dosyaya YAZILMAZ, cagri-zamaninda uretilir (bench/runner.py::
        compute_gt() ile ayni sozlesme)."""
        ann_path = pathlib.Path(self.cfg["paths"]["annotations_dir"]) / f"{seq_id}.txt"
        if not ann_path.exists():
            return {}
        frames = load_annotations(ann_path)
        m = self._manifest_data()[seq_id]
        out = {}
        for q, fn in self._queries().items():
            iv = fn(frames, m["n_frames"], m["fps"])
            if iv:
                out[q] = iv
        return out

    def window_params(self) -> dict:
        w = self.cfg["window"]
        return {"size_s": w["size_s"], "stride_s": w["stride_s"], "min_s": w["min_window_s"]}

    def license(self) -> str:
        return self.cfg["datasets"]["visdrone"]["license"]

    def manifest(self) -> DatasetManifest:
        seqs = self.list_sequences()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_version=self.cfg["datasets"]["visdrone"]["dataset_version"],
            source_hash=_SOURCE_ZIP_SHA256,
            split="train",
            item_count=len(seqs),
            query_count=len(self._queries()),
            retrieval_unit="video_interval",
            has_structured_filters=True,
            groundtruth_type="annotation_derived",
            embedding_cache_key=f"visdrone:{_SOURCE_ZIP_SHA256[:12]}",
        )


__all__ = ["VisDroneAdapter"]
