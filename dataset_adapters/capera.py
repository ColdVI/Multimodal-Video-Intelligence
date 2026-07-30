"""CapERA (Captioning Events in Aerial Videos - Bashmal ve ark. 2023, MDPI
Remote Sensing 15(8):2139, DOI 10.3390/rs15082139; ERA event-recognition
aerial video setinin (Mou/Hua ve ark. 2020) captioning katmani) icin somut
DatasetAdapter.

ONEMLI - split kimlik cakismasi: bu depoya kullanicidan gelen
capera_retrieval_pipeline.ipynb'nin kendi yorumu (hucre 5) acikca soyle
diyor: "video_id alone isn't guaranteed unique across train/test (same
filename can appear in both splits)" - bu yuzden video kimligi HER ZAMAN
split-nitelenmis: f"{split}__{video_id}" (ör. "train__Baseball_001.mp4").
Bu adapter AYNI semayi kullanir - farkli bir semaya gecmek, notebook'un
kendi gercek kosumundaki (artifacts/capera_validation.json, 2863/2864
video) sayilarla uyusmayan yeni bir kimlik uzayi yaratirdi.

VIDEO DOSYALARI BU DEPODA YOK: notebook videoyu Colab'in gecici diskine
(/content/Videos, Drive'a KAYDEDILMEYEN bir alan) aciyordu - bu depoda
yalniz caption JSON'lari ve all_results.json (agregatif sonuclar) var,
gercek .mp4 dosyalari YOK. load_video() bu yuzden VisDrone/MSRVTT
adapter'lariyla TUTARLI bir yol insa eder (var olup olmadigini caller
kontrol eder) ama var oldugunu iddia etmez; fps() ise gercek bir olcum
degil - video dosyasi olmadan olculemez, NotImplementedError firlatir
(uydurma bir sayi DONMEZ)."""
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

from .base import DatasetAdapter, DatasetManifest

# capera_retrieval_pipeline.ipynb hucre 0 (markdown): "every CapERA video
# is 5 seconds, so one video = one chunk" - notebook yazarinin kendi
# belgeledigi sabit; video dosyasi olmadigi icin bu depoda BAGIMSIZ
# dogrulanamadi, kaynagi acikca budur.
# Geriye donuk import uyumlulugu. Uretim loader'i bu degeri config.yaml'daki
# datasets.capera.fixed_duration_s alanindan okur.
FIXED_DURATION_S = 5.0

_CATEGORY_RE = re.compile(r"_\d+\.mp4$")

# capera_retrieval_pipeline.ipynb hucre 5 (kullanicinin kendi ERA_Dataset.zip
# cikarma kosumundan, os.listdir() ile DOGRULANMIS): ERA_Dataset.zip
# cikarilinca "Videos/Tra/<kategori>/..." ve "Videos/Test/<kategori>/..."
# olusuyor - "train"/"test" DEGIL. Ayrica gercek dosya adlarinda ".mp4"
# oncesinde fazladan bir BOSLUK var (ör. "Baseball_001 .mp4") - JSON'daki
# video_id'de ("Baseball_001.mp4") bu bosluk YOK.
_SPLIT_FOLDER_NAMES = {"train": "Tra", "test": "Test"}


def _real_video_filename(video_id: str) -> str:
    return video_id.replace(".mp4", " .mp4")


class CapERAAdapter(DatasetAdapter):
    dataset_id = "capera"
    name = "CapERA (ERA aerial event captioning)"
    has_captions = True

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._entries_cache = None

    def _paths(self) -> dict:
        c = self.cfg["datasets"]["capera"]
        return {"train": pathlib.Path(c["train_split"]), "test": pathlib.Path(c["test_split"])}

    def _entries(self) -> dict:
        """split-nitelenmis video kimligi -> {"captions", "category", "split"}
        - capera_retrieval_pipeline.ipynb hucre 5 ile AYNI kimliklendirme
        semasi (unique_key = f"{split}__{video_id}")."""
        if self._entries_cache is None:
            out = {}
            for split, path in self._paths().items():
                data = json.loads(path.read_text(encoding="utf-8"))
                for entry in data["ERA_caption"]:
                    video_id = entry["video_id"]
                    key = f"{split}__{video_id}"
                    out[key] = {
                        "captions": entry["annotation"]["English_caption"],
                        "category": _CATEGORY_RE.sub("", video_id),
                        "raw_video_id": video_id,
                        "split": split,
                    }
            self._entries_cache = out
        return self._entries_cache

    def list_sequences(self, split: str | None = None) -> list:
        entries = self._entries()
        if split is None:
            return sorted(entries)
        if split not in self._paths():
            raise ValueError(f"unknown CapERA split: {split}")
        return sorted(key for key, entry in entries.items() if entry["split"] == split)

    def entry(self, seq_id: str) -> dict:
        """Adapter tarafindan normalize edilmis tek kaydi guvenli kopya olarak dondurur."""
        entry = self._entries()[seq_id]
        return {**entry, "captions": list(entry["captions"])}

    def load_video(self, seq_id: str) -> pathlib.Path:
        """videos_dir, ERA_Dataset.zip'in cikarilmasiyla olusan 'Videos/'
        klasorunun KENDISI olmali - alt klasor adlari ('Tra'/'Test') ve
        dosya-adi bosluk kuraligi gercek zip cikisindan dogrulandi (yukarida)."""
        entry = self._entries()[seq_id]
        videos_dir = pathlib.Path(self.cfg["datasets"]["capera"]["videos_dir"])
        split_folder = _SPLIT_FOLDER_NAMES[entry["split"]]
        filename = _real_video_filename(entry["raw_video_id"])
        return videos_dir / split_folder / entry["category"] / filename

    def fps(self, seq_id: str) -> float:
        raise NotImplementedError(
            "CapERA video dosyalari bu depoda yok - fps gercek videodan "
            "olculmeden uydurulamaz. Sure biliniyor (FIXED_DURATION_S=5.0, "
            "notebook belgesi), fps bilinmiyor.")

    def ground_truth(self, seq_id: str) -> dict:
        """Her caption AYRI bir sorgu (notebook'un kendi manifest'i: video
        basina 5 caption = 5 sorgu satiri) - hepsi AYNI (0, 5.0) araligina
        isaret eder (retrieval_unit='video', VisDrone tarzi alt-interval
        lokalizasyonu YOK - bkz. DatasetAdapter.ground_truth docstring'i)."""
        entry = self._entries()[seq_id]
        duration_s = float(self.cfg["datasets"]["capera"].get("fixed_duration_s", FIXED_DURATION_S))
        return {cap: [(0.0, duration_s)] for cap in entry["captions"]}

    def captions(self, seq_id: str):
        return list(self._entries()[seq_id]["captions"])

    def window_params(self) -> dict:
        return {}

    def license(self) -> str:
        return self.cfg["datasets"]["capera"]["license"]

    def manifest(self, split: str | None = None) -> DatasetManifest:
        all_entries = self._entries()
        entries = all_entries if split is None else {
            key: value for key, value in all_entries.items() if value["split"] == split
        }
        if split is not None and split not in self._paths():
            raise ValueError(f"unknown CapERA split: {split}")
        paths = self._paths()
        source_bytes = (
            paths["test"].read_bytes() + paths["train"].read_bytes()
            if split is None else paths[split].read_bytes()
        )
        combined_hash = hashlib.sha256(source_bytes).hexdigest()
        n_queries = sum(len(e["captions"]) for e in entries.values())
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_version=(
                self.cfg["datasets"]["capera"].get("quality_dataset_version")
                if split is not None
                else self.cfg["datasets"]["capera"]["dataset_version"]
            ),
            source_hash=combined_hash,
            split=(split or "train+test (split-nitelenmis kimlik, bkz. modul docstring'i)"),
            item_count=len(entries),
            query_count=n_queries,
            retrieval_unit="video",
            has_structured_filters=False,
            groundtruth_type="caption_1to5",
            embedding_cache_key=f"capera:{combined_hash[:12]}",
        )


__all__ = ["CapERAAdapter", "FIXED_DURATION_S"]
