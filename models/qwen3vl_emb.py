"""Qwen/Qwen3-VL-Embedding-2B (Apache-2.0). Faz 4 adayi - MMEB-V2'de
sinifinin en iyisi olduğu iddia edilen, cok dilli (30+ dil, TR sorgu icin
ilgili), MRL (Matryoshka) destekli bir multimodal embedding modeli.

Video girdisi icin resmi bir sentence-transformers ornegi yok; bu adaptor
diger frame-average temelli baseline'imizla (models/siglip_avg.py) ayni
deseni kullanir: n_sample kare goruntu olarak ayri ayri kodlanir, ortalanir,
L2-normalize edilir. Bu bilinçli bir basitleştirmedir, native video-token
girdisi degildir - raporda acikca boyle isaretlenir.

MRL boyutlari ayri model kaydi olarak tanimlanir (qwen3vl_emb_2048/1024/
512/256) - şema boyutu tablo başına sabit oldugu icin (bkz. AGENTS.md model
ekleme prosedürü), boyutu runtime parametresi degil registry girişi yapmak
mimariyle tutarlı kalır ve boyut taraması otomatik bir bake-off satırına
dönüşür."""
import numpy as np
import torch

from common import offline_mode_enabled
from .base import VideoTextEmbedder

_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"


class _Qwen3VLEmbedderBase(VideoTextEmbedder):
    dim: int

    def __init__(self, device: str = None):
        # Agir bagimlilik (sentence-transformers + qwen-vl-utils) yalnizca
        # bu adaptor gercekten kullanildiginda import edilir - eksik/bozuk
        # olsa bile diger modelleri veya test suite'ini etkilemez.
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            _MODEL_ID, device=self.device, truncate_dim=self.dim,
            local_files_only=offline_mode_enabled(),
        )

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        return vec / np.linalg.norm(vec)

    def embed_video(self, frames: list, n_sample: int = 6) -> np.ndarray:
        # frame-average baseline (bkz. modul docstring'i) - PIL Image
        # bekleniyor; cv2 BGR ndarray gelirse RGB'ye cevir. 2B parametreli
        # modelin CPU maliyeti (~8.6sn/kare, gercek olculdu) yuzunden
        # SiglipAvg gibi az sayida kareye alt-orneklenir - 32 karenin
        # tumunu kodlamak pencere basina ~275sn'ye cikardi.
        idx = np.linspace(0, len(frames) - 1, min(n_sample, len(frames))).astype(int)
        frames = [frames[i] for i in idx]
        from PIL import Image
        images = []
        for f in frames:
            if isinstance(f, Image.Image):
                images.append(f)
            else:
                arr = np.asarray(f)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = arr[:, :, ::-1]  # BGR -> RGB varsayimi (cv2 kaynakli)
                images.append(Image.fromarray(arr))
        embs = self.model.encode([{"image": im} for im in images], convert_to_numpy=True)
        vec = embs.mean(axis=0)
        return vec / np.linalg.norm(vec)


class Qwen3VLEmb2048(_Qwen3VLEmbedderBase):
    name, dim = "qwen3vl_emb_2048", 2048


class Qwen3VLEmb1024(_Qwen3VLEmbedderBase):
    name, dim = "qwen3vl_emb_1024", 1024


class Qwen3VLEmb512(_Qwen3VLEmbedderBase):
    name, dim = "qwen3vl_emb_512", 512


class Qwen3VLEmb256(_Qwen3VLEmbedderBase):
    name, dim = "qwen3vl_emb_256", 256
