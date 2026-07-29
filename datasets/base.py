"""Farkli veri kaynaklarinin (VisDrone, arkadaslarin yukleyecegi datasetler,
CapERA/DVTMD/ERA gibi caption'li setler) ortak arayuzu. models/base.py'deki
VideoTextEmbedder deseninin ayni sekilde uygulanmasi - registry ile genisler,
mevcut kod degismez.

BU DOSYA YALNIZCA ARAYUZ TANIMIDIR. VisDrone'un kendisi ClickHouse ingest
hattinda (ingest/*.py) hala dogrudan config.yaml: paths.* okuyor - bu adapter
katmani ONUN YERINE gecmez, cross-dataset kod (bench/eval/artifact) icin
ORTAK bir okuma yuzeyi saglar (bkz. datasets/visdrone.py, datasets/msrvtt.py).
Bilincli kapsam karari: mevcut, dogrulanmis VisDrone ClickHouse davranisini
bozmadan, dataset-agnostik kodun (adaptive MRL harness, artifact contract)
hedefleyecegi somut bir sozlesme."""
import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class DatasetManifest:
    """Bir dataset'in tek-bakista kimligi - artifact'lara (bkz. artifacts/
    contract.py) gomulur ki hangi kosunun hangi veri/split/surumden
    uretildigi sonradan asla belirsiz olmasin."""
    dataset_id: str
    dataset_version: str
    source_hash: str
    split: str
    item_count: int
    query_count: int
    retrieval_unit: str  # ör. 'video_interval' | 'video'
    has_structured_filters: bool
    groundtruth_type: str  # ör. 'annotation_derived' | 'caption_1to1'
    embedding_cache_key: str


def qualified_id(dataset_id: str, video_id: str, t_start: float = None) -> tuple:
    """Ortak retrieval item kimligi: (dataset_id, video_id, t_start).
    Farkli dataset'lerde ayni video_id kullanilsa bile (ör. iki ayri setten
    ayni isimli klip) bu demet cakismaz - dataset_id ayirt edici. MSR-VTT
    gibi pencereleme yapmayan setlerde t_start=None (tum klip TEK birim)."""
    return (dataset_id, video_id, t_start)


class DatasetAdapter(ABC):
    dataset_id: str
    name: str
    has_captions: bool

    @abstractmethod
    def list_sequences(self) -> list:
        """Bu dataset'teki sekans/video kimlikleri."""

    @abstractmethod
    def load_video(self, seq_id: str) -> Path:
        """seq_id -> mp4 dosya yolu (kare dizini ise once video'ya cevrilmis olmali)."""

    @abstractmethod
    def fps(self, seq_id: str) -> float:
        """Sekansin gercek fps'i - manifest'ten okunur, sabit deger varsayilmaz."""

    @abstractmethod
    def ground_truth(self, seq_id: str) -> dict:
        """sorgu -> zaman araliklari. Anotasyondan mi caption'dan mi turedigini
        eval raporu acikca belirtmeli (protokoller birbirinden farkli)."""

    def captions(self, seq_id: str):
        """Varsa caption listesi, yoksa None. has_captions=False ise override
        etmeye gerek yok, varsayilan None döner."""
        return None

    def window_params(self) -> dict:
        """size_s/stride_s/min_s override - dataset'e ozgu klip suresine gore
        (ör. CapERA ~5sn klipler, VisDrone'un 8s/4s penceresine uymuyor)."""
        return {}

    @abstractmethod
    def license(self) -> str:
        """SPDX benzeri kisa lisans etiketi (ör. 'MIT', 'CC-BY-NC-SA-4.0').
        Ticari kullanim kontrolu bu deger uzerinden yapilir - bos/None kabul
        edilmez, bilinmiyorsa acikca 'UNKNOWN' donsun."""

    @abstractmethod
    def manifest(self) -> DatasetManifest:
        """Bu dataset kosumunun DatasetManifest'i - artifact yazarken
        (artifacts/search_runs/<run_id>/dataset_manifest.json) oldugu gibi
        gomulur."""
