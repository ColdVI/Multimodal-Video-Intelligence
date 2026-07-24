"""Kare-ortalama SigLIP2 - mevcut (uretim) sistemin proxy'si / baseline.

Checkpoint'in resmi config'i `model_type: siglip` kullanir ve model karti
dogrudan AutoModel onerir. Adi SigLIP2 olsa da Siglip2Model'e zorlamak patch
embedding sekillerini bozar; gercek yuklemede yakalanmistir.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from common import offline_mode_enabled
from .base import VideoTextEmbedder, feature_tensor


class SiglipAvg(VideoTextEmbedder):
    name, dim = "siglip2_frameavg", 1152

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "google/siglip2-so400m-patch14-384"
        local_files_only = offline_mode_enabled()
        self.proc = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).to(self.device).eval()

    @torch.no_grad()
    def embed_video(self, frames: list) -> np.ndarray:
        idx = np.linspace(0, len(frames) - 1, 8).astype(int)
        images = [frames[i] for i in idx]
        inputs = self.proc(images=images, return_tensors="pt").to(self.device)
        feats = feature_tensor(self.model.get_image_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1).mean(0)
        return torch.nn.functional.normalize(feats, dim=0).cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.proc(text=[text], return_tensors="pt",
                            padding="max_length").to(self.device)
        feats = feature_tensor(self.model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].cpu().numpy()
