"""Model registry. get_embedder(name) cagrildiginda ornek yoksa olusturur,
varsa onbellekten doner (agir model yuklemesi tekrar tetiklenmesin diye)."""
from common import configure_runtime_environment

configure_runtime_environment()

from .base import VideoTextEmbedder
from .siglip_avg import SiglipAvg
from .xclip_hf import XClipHF

_REGISTRY = {
    SiglipAvg.name: SiglipAvg,
    XClipHF.name: XClipHF,
}
_instances = {}


def get_embedder(name: str, device: str = None) -> VideoTextEmbedder:
    if name not in _instances:
        if name not in _REGISTRY:
            raise KeyError(
                f"Bilinmeyen embedder '{name}'. Mevcut: {sorted(_REGISTRY)}. "
                f"Yeni model eklemek icin: models/<isim>.py yaz, buradaki "
                f"_REGISTRY'ye ekle, schema.sql'e clips_<isim> tablosu ekle."
            )
        _instances[name] = _REGISTRY[name](device=device)
    return _instances[name]


def available_models() -> list:
    return sorted(_REGISTRY)
