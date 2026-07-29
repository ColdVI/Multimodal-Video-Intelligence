"""Run manifest yazicisi (spec SS9.4): her notebook kosumunun baglamini
(hardware_profile dahil) JSON'a yazar. bench/spec.py::RunSpec'in
run_id/hardware_profile deseniyle ayni mantik - "hangi donanimda olculdu"
sorusu her artifact'ta zorunlu alan olarak durur, sonradan tahmin edilmez."""
import dataclasses
import datetime
import json
import pathlib

import torch


@dataclasses.dataclass(frozen=True)
class RunManifest:
    notebook: str
    hardware_profile: str  # "colab-T4-fp16-sdpa" | "colab-A100-bf16-fa2" | "local-cpu" | ...
    dataset_id: str = ""
    model_id: str = ""
    model_revision: str = ""
    dtype: str = ""
    attn_implementation: str = ""
    frames_per_item: int = None
    extra: dict = dataclasses.field(default_factory=dict)
    generated_at: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def detect_hardware_profile() -> dict:
    """SS3.2(c): Turing/T4 profilinde fp16+sdpa, Ampere+/A100/L4'te
    bf16+flash_attention_2. CUDA yoksa 'local-cpu' - notebook 02'nin GPU
    kapisi bunu ayrica kontrol eder, bu fonksiyon sadece SAPTAR, karar
    vermez (raise etmez)."""
    if not torch.cuda.is_available():
        return {
            "hardware_profile": "local-cpu",
            "gpu_name": None,
            "cuda_available": False,
            "torch_version": torch.__version__,
        }
    name = torch.cuda.get_device_name(0)
    major, _minor = torch.cuda.get_device_capability(0)
    # Turing = sm_75 (major=7, T4 dahil); Ampere+ = major>=8 (A100/L4 dahil).
    if major >= 8:
        profile = f"colab-{name.replace(' ', '')}-bf16-fa2"
    else:
        profile = f"colab-{name.replace(' ', '')}-fp16-sdpa"
    return {
        "hardware_profile": profile,
        "gpu_name": name,
        "cuda_available": True,
        "torch_version": torch.__version__,
        "compute_capability": f"{major}.{_minor}",
    }


def write_manifest(manifest: RunManifest, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.notebook}_manifest.json"
    out_path.write_text(json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    return out_path


__all__ = ["RunManifest", "detect_hardware_profile", "write_manifest"]
