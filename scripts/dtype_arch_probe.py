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
import warnings

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

    # SIRALAMA ONEMLI: nn.Module.half() YERINDE (in-place) mutasyon yapar ve
    # self dondurur (dogrulandi: `m.half() is m` -> True) - model_fp16 ve
    # model AYNI nesne. torch.compile testi bu satirdan SONRA calisirsa
    # aslinda "fp16 + compile" olcer, "native dtype + compile" degil (bu
    # bug'in kendisi bir Colab kosumunda yakalandi, once compile test edilip
    # DAHA SONRA fp16'ya donusum yapiliyor, boylece her ikisi de dogru
    # dtype'ini olcuyor).
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

    if device == "cuda":
        try:
            model_fp16 = model.half()  # YIKICI: model'i yerinde fp16'ya cevirir
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

    return result


def _make_test_image(image_size: int):
    from PIL import Image
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8))


def _load_fresh_model(model_id: str, device: str, dtype_label: str):
    """HER cagrida BRAND NEW bir model nesnesi doner - nn.Module.half()
    yerinde mutasyon yapip self dondurdugu icin (dogrulandi: m.half() is m
    -> True), olcum hucreleri arasinda paylasilan/mutasyona ugramis bir
    model kullanmak bir onceki hucrenin durumunu sizdirir. dtype_label:
    'native_bf16' (modelin kendi varsayilani) veya 'fp16' (.half() ile
    zorlanmis)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device)
    if dtype_label == "fp16":
        model = model.half()
    return model


def probe_sdpa_backend_matrix(model_id: str, image_size: int, skip: bool = False) -> dict:
    """dtype {native_bf16, fp16} x SDPA backend {MATH, EFFICIENT_ATTENTION,
    FLASH_ATTENTION} - 2x3 matris. Her (dtype, backend) hucresi icin backend
    torch.nn.attention.sdpa_kernel ile ZORLANIR; desteklenmiyorsa RuntimeError
    aliniyor demektir (_time_calls bunu yakalayip {"error": ...} donduruyor,
    kod cokmuyor). Reddedilen hucreler icin PyTorch'un urettigi UserWarning
    da (varsa) yakalanip raporlanir.

    NEDEN ONEMLI: attn_implementation='sdpa' bir HuggingFace CONFIG etiketi,
    calisan kernel degil - SDPA bir dispatcher, flash/mem_efficient/math
    arasindan runtime'da dtype+donanima gore secim yapiyor. "Fark saf
    dtype'tan geliyor, backend'den degil" cikarimi bu yuzden kanitlanmamis -
    dtype backend secimini BELIRLEYEBILIYOR. Bu matris ikisini ayristirir."""
    if skip:
        return {"skipped": "--skip-sdpa-matrix"}

    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = _make_test_image(image_size)
    backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]

    matrix = {}
    for dtype_label in ["native_bf16", "fp16"]:
        try:
            model = _load_fresh_model(model_id, device, dtype_label)
        except Exception as exc:
            for backend in backends:
                matrix[f"{dtype_label}/{backend.name}"] = {"error": f"model yuklenemedi: {exc}"}
            continue

        for backend in backends:
            def encode_once(m=model, b=backend):
                with sdpa_kernel(b):
                    m.encode([{"image": image}], convert_to_numpy=True)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                timing = _time_calls(encode_once, warmup=2, n=5)
            entry = dict(timing)
            if caught:
                entry["pytorch_warnings"] = [str(w.message) for w in caught]
            matrix[f"{dtype_label}/{backend.name}"] = entry

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return matrix


def decompose_speedup(matrix: dict) -> dict:
    """68.81x gibi tek bir sayiyi iki bilesene ayirmaya calisir:
    - attention_backend_ratio: AYNI dtype icinde en yavas/en hizli backend orani
    - gemm_dtype_ratio: AYNI backend icinde bf16 vs fp16 orani
    Hangi hucrelerin gercekten calisacagi GPU'ya gore degisebilecegi icin
    sabit tek bir "attention payi X, dtype payi Y" formulu YAZILMIYOR -
    olculebilen tum anlamli ikili karsilastirmalar raporlanir, yorumu
    insan/rapor asamasinda yapilir."""
    working = {k: v["median_s"] for k, v in matrix.items()
              if isinstance(v, dict) and "median_s" in v}
    comparisons = {}

    for dtype_label in ["native_bf16", "fp16"]:
        cells = {k: v for k, v in working.items() if k.startswith(dtype_label + "/")}
        if len(cells) >= 2:
            slowest_key = max(cells, key=cells.get)
            fastest_key = min(cells, key=cells.get)
            comparisons[f"attention_backend_ratio_{dtype_label}"] = {
                "slowest": slowest_key, "slowest_s": cells[slowest_key],
                "fastest": fastest_key, "fastest_s": cells[fastest_key],
                "ratio_x": round(cells[slowest_key] / cells[fastest_key], 2),
            }

    for backend_name in ["MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION"]:
        bf16_key, fp16_key = f"native_bf16/{backend_name}", f"fp16/{backend_name}"
        if bf16_key in working and fp16_key in working:
            comparisons[f"gemm_dtype_ratio_{backend_name}"] = {
                "bf16_s": working[bf16_key], "fp16_s": working[fp16_key],
                "ratio_x": round(working[bf16_key] / working[fp16_key], 2),
            }
    return comparisons


def probe_compile_matrix(model_id: str, image_size: int, skip: bool = False) -> dict:
    """bf16 -> bf16+compile -> fp16 -> fp16+compile, HER adimda TEMIZ
    (yeniden yuklenmis) bir model ile. probe_model()'deki tekil compile
    testinden farki: burada 4 hucrenin HICBIRI bir onceki hucrenin
    mutasyona ugramis modelini paylasmiyor - tam izolasyon.

    BEKLENTI (olculecek, varsayilmayacak): TorchInductor bf16'yi yalniz
    sm_80+ icin destekliyor (pytorch#118122) - T4 (sm_75) + bf16 + compile
    ya BackendCompilerFailed atar ya da bir alt-grafta eager'a duser ve
    kazanc ~0 olur."""
    if skip:
        return {"skipped": "--skip-compile"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = _make_test_image(image_size)
    results = {}

    for dtype_label in ["native_bf16", "fp16"]:
        for use_compile in [False, True]:
            key = f"{dtype_label}{'_compiled' if use_compile else ''}"
            try:
                model = _load_fresh_model(model_id, device, dtype_label)
                target = torch.compile(model) if use_compile else model

                def encode_once(t=target):
                    t.encode([{"image": image}], convert_to_numpy=True)

                results[key] = _time_calls(encode_once, warmup=(3 if use_compile else 1), n=5)
            except Exception as exc:
                results[key] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return results


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


def print_sdpa_decomposition(matrix: dict, decomposition: dict, compile_matrix: dict):
    print("\n=== SDPA backend matrisi (dtype x backend) ===")
    for key, v in matrix.items():
        if not isinstance(v, dict):
            print(f"  {key}: {v}")
        elif "error" in v:
            print(f"  {key}: REDDEDILDI/HATA - {v['error'][:150]}")
        else:
            warn = f" [uyari: {v['pytorch_warnings']}]" if v.get("pytorch_warnings") else ""
            print(f"  {key}: medyan={v['median_s']:.4f}s p95={v['p95_s']:.4f}s{warn}")

    print("\n=== 68.81x ayristirmasi ===")
    if not decomposition:
        print("  Yeterli hucre basarili olmadi, ayristirma yapilamadi.")
    for key, v in decomposition.items():
        print(f"  {key}: {v}")

    print("\n=== torch.compile matrisi (bf16/fp16 x compile acik/kapali, TEMIZ yeniden yukleme) ===")
    for key, v in compile_matrix.items():
        if not isinstance(v, dict):
            print(f"  {key}: {v}")
        elif "error" in v:
            print(f"  {key}: HATA - {v['error'][:150]}")
        else:
            print(f"  {key}: medyan={v['median_s']:.4f}s p95={v['p95_s']:.4f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--skip-compile", action="store_true",
                    help="torch.compile A/B'sini atla (yavas/kararsiz olabilir)")
    ap.add_argument("--sdpa-decomposition", action="store_true",
                    help="68.81x'i attention-backend payi ve GEMM-dtype payina "
                         "ayiran ek olcumu de calistir (model_bakeoff_ve_demo_paketi.md "
                         "Bolum 4). Ek olarak ~8 model yuklemesi gerektirir, "
                         "birkac dakika daha surer.")
    ap.add_argument("--out", default=None,
                    help="varsayilan: artifacts/dtype_arch_probe_<gpu-adi>.json")
    args = ap.parse_args()

    hw = probe_hardware()
    model_result = probe_model(args.model, args.image_size, args.skip_compile)
    print_summary(hw, model_result)

    gpu_slug = hw.get('gpu_name', 'unknown').replace(' ', '_')
    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(
        f"artifacts/dtype_arch_probe_{gpu_slug}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hardware": hw, "model": model_result}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nJSON kanit: {out_path}")

    if args.sdpa_decomposition:
        matrix = probe_sdpa_backend_matrix(args.model, args.image_size)
        decomposition = decompose_speedup(matrix) if isinstance(matrix, dict) and "skipped" not in matrix else {}
        compile_matrix = probe_compile_matrix(args.model, args.image_size, args.skip_compile)
        print_sdpa_decomposition(matrix, decomposition, compile_matrix)

        sdpa_out_path = pathlib.Path(f"artifacts/sdpa_decomposition_{gpu_slug}.json")
        sdpa_out_path.write_text(json.dumps({
            "hardware": hw, "model_id": args.model,
            "sdpa_backend_matrix": matrix,
            "speedup_decomposition": decomposition,
            "compile_matrix": compile_matrix,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSDPA ayristirma JSON kaniti: {sdpa_out_path}")


if __name__ == "__main__":
    main()
