"""Faz 6 arastirma spec'i (07_MRL_VE_VECTOR_BACKEND_ARASTIRMA_SPEC.md), SS8:
tum notebook'larin okudugu TEK config. Buradaki degerler spec'te "degisim
adaleti - sabitler" olarak kilitlendi; notebook'lar bunu import eder, kendi
sabitlerini tanimlamaz."""
import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True)
class ResearchConfig:
    warmup: int = 3
    measured_repetitions: int = 10
    top_k: int = 10
    frames_per_item: int = 8
    window_size_s: float = 8.0
    stride_s: float = 4.0
    random_seed: int = 42
    gap_factor: int = 10  # AU-AIR video-sinir yeniden kurulumu (SS4.2 adim 4)
    dims: tuple = (2048, 1024, 512, 256)
    adaptive_base_dims: tuple = (256, 512)
    adaptive_top_n_sweep: tuple = (20, 50, 100, 200)
    statistical_power_threshold: int = 150  # SS12: 150 sorguya ulasana kadar baglayici degil
    query_instruction: str = "Retrieve the aerial video segment described by the query."

    research_root: pathlib.Path = pathlib.Path("artifacts/research")
    data_root: pathlib.Path = pathlib.Path("data/research")

    def artifact_path(self, name: str) -> pathlib.Path:
        return self.research_root / name


DEFAULT = ResearchConfig()

__all__ = ["ResearchConfig", "DEFAULT"]
