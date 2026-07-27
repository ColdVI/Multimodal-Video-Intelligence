"""Farkli veri kaynaklarinin (VisDrone, arkadaslarin yukleyecegi datasetler,
CapERA/DVTMD/ERA gibi caption'li setler) ortak arayuzu. models/base.py'deki
VideoTextEmbedder deseninin ayni sekilde uygulanmasi - registry ile genisler,
mevcut kod degismez.

BU DOSYA YALNIZCA ARAYUZ TANIMIDIR. VisDrone'un kendisi bu arayuze henuz
tasinmadi (ingest/*.py hala dogrudan config.yaml: paths.* okuyor) - bu
bilincli bir kapsam karari: mevcut, dogrulanmis VisDrone davranisini
bozmadan, yeni dataset eklemek isteyenlerin (bkz. scripts/inspect_dataset.py)
hedefleyecegi somut bir sozlesme olsun diye once arayuz kuruldu."""
from abc import ABC, abstractmethod
from pathlib import Path


class DatasetAdapter(ABC):
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
