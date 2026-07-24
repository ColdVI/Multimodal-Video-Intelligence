"""Tum embedding modelleri bu arayuzu uygular. Yeni model eklemek = yeni
dosya + models/__init__.py'deki registry'e bir satir."""
from abc import ABC, abstractmethod

import numpy as np


def feature_tensor(output):
    """Transformers 4.x Tensor ve 5.x pooled ModelOutput dönüşlerini birler."""
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if hasattr(output, "shape"):
        return output
    raise TypeError(
        "Model feature çıktısı Tensor veya pooler_output içeren bir nesne olmalı; "
        f"gelen tip: {type(output).__name__}"
    )


class VideoTextEmbedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_video(self, frames: list) -> np.ndarray:
        """8sn pencereden ornekleneknis RGB kareler -> tek L2-normalize vektor."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Sorgu metni -> ayni uzayda L2-normalize vektor."""
