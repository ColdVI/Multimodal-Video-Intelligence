"""HuggingFace'teki microsoft/xclip-base-patch16-zero-shot.

DIKKAT: Bu, Ni ve ark. (Kinetics video siniflandirma) X-CLIP'idir - raporun
onerdigi, Ma ve ark. (2022, ACM MM, AOSM, retrieval-ozel) X-CLIP'i DEGIL.
Bake-off'ta ikisi ayri satir olmali; bu adapter yalnizca ilk haftada boru
hattini acmak icin pragmatik bir baslangictir (bkz. CONTEXT.md).

API dogrulandi: processor(videos=..., return_tensors="pt") ve
model.get_video_features(**inputs) transformers'in kendi kaynak kodundaki
kullanim ornegiyle birebir ayni
(src/transformers/models/x_clip/modeling_x_clip.py, huggingface/transformers
main branch). Hic calistirilmadi (bu sandboxta huggingface.co erisimi yok) -
yalniz kaynak koduna karsi API yuzeyi dogrulandi.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from .base import VideoTextEmbedder, feature_tensor


class XClipHF(VideoTextEmbedder):
    name, dim = "xclip_hf_zeroshot", 512

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "microsoft/xclip-base-patch16-zero-shot"
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def embed_video(self, frames: list) -> np.ndarray:
        idx = np.linspace(0, len(frames) - 1, 32).astype(int)
        clip = [frames[i] for i in idx]
        inputs = self.proc(videos=[clip], return_tensors="pt").to(self.device)
        feats = feature_tensor(self.model.get_video_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.proc(text=[text], return_tensors="pt",
                            padding=True).to(self.device)
        feats = feature_tensor(self.model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].cpu().numpy()
