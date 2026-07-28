"""Karsilastirma istatistikleri: paired bootstrap guven araligi ve
saptanabilir minimum fark (MDE). "X modeli Y'den iyi" yerine "fark =
a [lo, hi], MDE = c" formatini destekler (bkz. TASKS.md Faz 6,
recall@1'in n_gt tarafindan domine edilmesi bulgusu - bu modul o
sorunun genel cozumu: kucuk/gurultu icindeki farklari guven araligiyla
raporlamak, tek sayiyla "kazandi" dememek)."""
import math


def paired_bootstrap_ci(values_a: list, values_b: list, n_resamples: int = 2000,
                        ci: float = 0.95, seed: int = 0) -> dict:
    """values_a/values_b: AYNI sorgu sirasinda iki modelin/kosulun olctugu
    metrik degerleri (ör. iki modelin ayni 28 sorguda nDCG@10'u) - paired,
    yani values_a[i] ve values_b[i] AYNI sorguya ait olmali.

    Donus: {"mean_diff", "ci_lo", "ci_hi", "cohens_d", "n"}. Fark pozitifse
    a > b demektir. Guven araligi 0'i iceriyorsa fark istatistiksel olarak
    ayirt edilemez (bunu cagiran taraf yorumlamali, bu fonksiyon karar
    vermez)."""
    if len(values_a) != len(values_b):
        raise ValueError(f"paired veri gerekli: len(a)={len(values_a)} != len(b)={len(values_b)}")
    n = len(values_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "cohens_d": 0.0, "n": 0}

    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean_diff = sum(diffs) / n

    if n == 1:
        return {"mean_diff": mean_diff, "ci_lo": mean_diff, "ci_hi": mean_diff,
               "cohens_d": 0.0, "n": 1}

    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std = variance ** 0.5
    cohens_d = mean_diff / std if std > 0 else 0.0

    # basit (kutuphanesiz) bootstrap: her resample'da n indeks yerine-koyarak
    # secilir, ortalama fark hesaplanir. numpy'siz calisir (bu modul repo'da
    # her yerde import edilebilir olsun diye agir bagimlilik eklemiyor).
    import random
    rng = random.Random(seed)
    resample_means = []
    for _ in range(n_resamples):
        resampled_diffs = [diffs[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(resampled_diffs) / n)
    resample_means.sort()

    alpha = 1 - ci
    lo_idx = max(0, int((alpha / 2) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))

    return {
        "mean_diff": mean_diff,
        "ci_lo": resample_means[lo_idx],
        "ci_hi": resample_means[hi_idx],
        "cohens_d": cohens_d,
        "n": n,
    }


# Standart normal dagilim ters-CDF (quantile) icin kucuk bir yaklasim
# tablosu yerine Acklam algoritmasinin basitlestirilmis hali - scipy
# bagimliligi eklemeden z-skoru hesaplamak icin.
def _norm_ppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p (0,1) araliginda olmali")
    # Beasley-Springer-Moro yaklasimi (yeterli hassasiyet, ek bagimlilik yok)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - p_low:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
          (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def minimum_detectable_effect(n: int, std: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Standart iki-orneklem MDE formulu: MDE = (z_(alpha/2) + z_power) * std * sqrt(2/n).
    n: sorgu sayisi, std: metrigin gozlenen standart sapmasi. Kapali-form,
    simulasyon gerektirmez - "bu set 0.XX altindaki farklari ayirt edemez"
    cumlesinin kaynagi budur."""
    if n <= 0:
        return float("inf")
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_power = _norm_ppf(power)
    return (z_alpha + z_power) * std * math.sqrt(2 / n)
