"""GPU zorunlulugu kapisi: agir embedding modelleri (Qwen3-VL-Embedding-2B
gibi) bu makinede CPU'da calisirsa saatler/gunler surer ve genelde yari
yolda birakilir - ortaya "nerede, ne kadar surede uretildigi belirsiz" bir
artifact cikar. Bu modul o durumu sessizce olusturmak yerine acikca durdurur.

Gercek olcum (TASKS.md, "Kritik CPU-maliyet bulgusu"): Qwen3-VL-Embedding-2B
bu depodaki gelistirme CPU'sunda 73 pencere icin 1062 dakika surdu -
pencere basina ~14.5 dakika (870 sn). L4 GPU'da (artifacts/
colab_gpu_bench_l4_v2.json, qwen3vl_emb_2048 embed_video mean_s) pencere
basina 2.443 sn - yaklasik 356x fark. xclip_hf_zeroshot ve siglip2_frameavg
bu kapinin KAPSAMI DISINDA: ikisi de bu CPU'da defalarca calisti ve
dakikalar mertebesinde kaldi (bkz. STATUS.md) - kapi yalniz Qwen ailesi
icin gecerli, script'in tamamini degil."""
import torch

# TASKS.md "Kritik CPU-maliyet bulgusu": 73 pencere / 1062 dakika, bu CPU'ya
# ozgu gercek olcum - yuvarlanmis "~14.5 dk/pencere" yerine tam degerden
# turetildi ki n_windows=73 verildiginde belgedeki 17.7 saati birebir uretsin.
QWEN_CPU_S_PER_WINDOW = 1062 * 60 / 73  # ~872.9 sn/pencere
# artifacts/colab_gpu_bench_l4_v2.json: qwen3vl_emb_2048 timing.embed_video.mean_s (gercek L4 kosumu).
QWEN_GPU_S_PER_WINDOW = 2.443


def require_gpu(task: str, cpu_estimate_h: float, gpu_estimate_min: float) -> None:
    """GPU yoksa acikca durur. Tahminleri cagiran taraf hesaplar (bu
    fonksiyon tahmin uretmez) - sadece kapiyi uygular ve mesaji
    standartlastirir. CPU'da sessizce gunlerce surecek bir kosum
    baslatmamak icin SystemExit kullanir (import zamaninda degil, cagri
    zamaninda kontrol eder ki testler torch.cuda.is_available'i mock
    edebilsin)."""
    if not torch.cuda.is_available():
        raise SystemExit(
            f"[GPU KAPISI] {task} GPU gerektiriyor.\n"
            f"  GPU'da tahmini : {gpu_estimate_min:.0f} dk\n"
            f"  CPU'da tahmini : {cpu_estimate_h:.1f} saat -> IPTAL\n"
            f"  Colab GPU oturumu acip tekrar deneyin."
        )


def require_gpu_for_qwen_windows(task: str, n_windows: int) -> None:
    """qwen3vl_emb ailesi icin n_windows'tan (pencere veya video-klip
    sayisi) tahmini sureyi gercek olculmus per-window hizlardan hesaplayip
    require_gpu'yu cagirir. VisDrone disi kullanimlarda (ör. MSR-VTT tum-
    klip embedleme) bu VisDrone pencere hizindan turetilen KABA bir
    tahmindir - kesin degil, sadece kapiyi tetiklemeye yeter."""
    cpu_estimate_h = n_windows * QWEN_CPU_S_PER_WINDOW / 3600
    gpu_estimate_min = n_windows * QWEN_GPU_S_PER_WINDOW / 60
    require_gpu(task, cpu_estimate_h, gpu_estimate_min)


__all__ = ["require_gpu", "require_gpu_for_qwen_windows",
          "QWEN_CPU_S_PER_WINDOW", "QWEN_GPU_S_PER_WINDOW"]
