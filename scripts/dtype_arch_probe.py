"""P0-C: bir modelin GPU'da dtype/attention/derleme yolunun GERCEKTEN
desteklenip desteklenmedigini olcer - "muhtemelen bf16/Turing sorunu"
gibi cikarimsal iddialari olcume cevirir.

BAGLAM: Qwen3-VL-Embedding-2B pencere basina T4'te 341.42s, L4'te 2.44s
olculdu (139.8x). Ayni makine degisikliginde X-CLIP/SigLIP2 yalniz ~2.3x
hizlandi - yani fark Qwen'e ozgu, genel GPU nesli farki degil. Hipotez:
T4 (Turing, sm_75) bf16'yi tensor-core'da desteklemiyor, FlashAttention-2
sm_80+ istiyor, TorchInductor bf16'yi yalniz sm_80+ icin derliyor. Bu
script hipotezi dogrudan olcume cevirir.

DIKKAT: torch.cuda.is_bf16_supported() sm_75'te (Turing/Volta) de True
donebilir - bu fonksiyona guvenmeyin, karari dogrudan compute capability
(>= (8,0)) ile verin.

Tek basina calisir (bu repoyu import etmez), model adi parametre.

Kullanim:
    python scripts/dtype_arch_probe.py --model Qwen/Qwen3-VL-Embedding-2B
    python scripts/dtype_arch_probe.py --model Qwen/Qwen3-VL-Embedding-2B --skip-compile
"""
import argparse
import json
import pathlib
import platform
import time

import numpy as np
import torch


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_calls(fn, warmup=3, n=10):
    """warmup kez calistirir (zamanlanmaz), sonra n kez olcer. OOM'u
    yakalar, coker degil - {"error": ...} doner."""
    try:
        for _ in range(warmup):
            fn()
        _synchronize()
        durations = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            _synchronize()
            durations.append(time.perf_counter() - t0)
        durations.sort()
        mid = len(durations) // 2
        median = durations[mid] if len(durations) % 2 else (durations[mid - 1] + durations[mid]) / 2
        p95_idx = min(int(len(durations) * 0.95), len(durations) - 1)
        return {"median_s": median, "p95_s": durations[p95_idx], "n": n, "raw_s": durations}
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM: {exc}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def probe_hardware() -> dict:
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        info["warning"] = "CUDA yok - bu probe GPU icin tasarlandi, CPU'da mimari-uyum bilgisi anlamsiz."
        return info

    cc = torch.cuda.get_device_capability(0)
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["compute_capability"] = list(cc)
    info["cuda_version"] = torch.version.cuda
    info["is_bf16_supported_reported"] = torch.cuda.is_bf16_supported()
    info["is_bf16_supported_reported_WARNING"] = (
        "torch.cuda.is_bf16_supported() Turing/Volta'da (sm<80) da True "
        "donebilir - buna guvenmeyin, asagidaki bf16_native_by_capability "
        "alanina bakin.")
    info["bf16_native_by_capability"] = cc >= (8, 0)
    info["flash_attention_2_supported_by_capability"] = cc >= (8, 0)
    try:
        import flash_attn  # noqa: F401
        info["flash_attn_importable"] = True
    except Exception as exc:
        info["flash_attn_importable"] = False
        info["flash_attn_import_error"] = str(exc)
    return info


def probe_model(model_id: str, image_size: int, skip_compile: bool) -> dict:
    from sentence_transformers import SentenceTransformer
    from PIL import Image

    result = {"model_id": model_id}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    model = SentenceTransformer(model_id, device=device)
    result["load_s"] = round(time.perf_counter() - t0, 2)

    try:
        result["native_dtype"] = str(next(model.parameters()).dtype)
    except Exception as exc:
        result["native_dtype"] = f"okunamadi: {exc}"

    attn_impl = None
    for sub in getattr(model, "_modules", {}).values():
        auto_model = getattr(sub, "auto_model", None)
        config = getattr(auto_model, "config", None)
        if config is not None:
            attn_impl = getattr(config, "_attn_implementation", None)
            break
    result["attn_implementation"] = attn_impl

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8))

    def encode_once():
        model.encode([{"image": image}], convert_to_numpy=True)

    result["native_dtype_timing"] = _time_calls(encode_once)

    if device == "cuda":
        try:
            model_fp16 = model.half()
            def encode_fp16():
                model_fp16.encode([{"image": image}], convert_to_numpy=True)
            result["fp16_timing"] = _time_calls(encode_fp16)
            if "error" not in result["fp16_timing"] and "error" not in result["native_dtype_timing"]:
                result["fp16_speedup_x"] = round(
                    result["native_dtype_timing"]["median_s"] / result["fp16_timing"]["median_s"], 2)
        except Exception as exc:
            result["fp16_timing"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        result["fp16_timing"] = {"skipped": "CUDA yok"}

    if skip_compile:
        result["torch_compile_timing"] = {"skipped": "--skip-compile"}
    else:
        try:
            compiled = torch.compile(model)
            def encode_compiled():
                compiled.encode([{"image": image}], convert_to_numpy=True)
            result["torch_compile_timing"] = _time_calls(encode_compiled, warmup=1, n=5)
        except Exception as exc:
            result["torch_compile_timing"] = {"error": f"{type(exc).__name__}: {exc}"}

    return result


def print_summary(hw: dict, model_result: dict):
    print("\n=== Donanim ===")
    for k, v in hw.items():
        print(f"  {k}: {v}")
    print("\n=== Model ===")
    print(f"  model_id: {model_result['model_id']}")
    print(f"  native_dtype: {model_result.get('native_dtype')}")
    print(f"  attn_implementation: {model_result.get('attn_implementation')}")
    nt = model_result.get("native_dtype_timing", {})
    if "error" in nt:
        print(f"  native dtype timing: HATA - {nt['error']}")
    else:
        print(f"  native dtype median: {nt.get('median_s', 0):.3f}s (p95={nt.get('p95_s', 0):.3f}s)")
    ft = model_result.get("fp16_timing", {})
    if "error" in ft:
        print(f"  fp16 timing: HATA - {ft['error']}")
    elif "skipped" in ft:
        print(f"  fp16 timing: atlandi ({ft['skipped']})")
    else:
        print(f"  fp16 median: {ft.get('median_s', 0):.3f}s (p95={ft.get('p95_s', 0):.3f}s)")
        if "fp16_speedup_x" in model_result:
            print(f"  fp16 hizlanma: {model_result['fp16_speedup_x']}x")
    ct = model_result.get("torch_compile_timing", {})
    if "error" in ct:
        print(f"  torch.compile: HATA - {ct['error']}")
    elif "skipped" in ct:
        print(f"  torch.compile: atlandi ({ct['skipped']})")
    else:
        print(f"  torch.compile median: {ct.get('median_s', 0):.3f}s (p95={ct.get('p95_s', 0):.3f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--skip-compile", action="store_true",
                    help="torch.compile A/B'sini atla (yavas/kararsiz olabilir)")
    ap.add_argument("--out", default=None,
                    help="varsayilan: artifacts/dtype_arch_probe_<gpu-adi>.json")
    args = ap.parse_args()

    hw = probe_hardware()
    model_result = probe_model(args.model, args.image_size, args.skip_compile)
    print_summary(hw, model_result)

    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(
        f"artifacts/dtype_arch_probe_{hw.get('gpu_name', 'unknown').replace(' ', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hardware": hw, "model": model_result}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nJSON kanit: {out_path}")


if __name__ == "__main__":
    main()
