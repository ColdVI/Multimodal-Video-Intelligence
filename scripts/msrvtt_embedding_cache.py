"""MSR-VTT video embedding onbellegi (Faz 6 GPU-hazirlik). Pahali (Qwen,
GPU-gated, ~41dk/1000video L4'te) video embed'lerini kalici diske yazar -
YARIM KALAN bir kosumda zaten tamamlanmis videolar TEKRAR EMBED EDILMEZ.
cache_key modelin/verinin TAM kimligini kapsar - herhangi bir alan
degisirse (model revision, n_sample, dataset hash...) FARKLI bir dosyaya
yazilir, eski/gecersiz sonuclarla sessizce karismaz."""
import hashlib
import json
import pathlib


def cache_key(dataset_id: str, split: str, dataset_hash: str, model_id: str,
             model_revision: str, n_sample: int, frame_sampling_version: str,
             dtype: str, dimension_source: str) -> str:
    payload = "|".join([dataset_id, split, dataset_hash, model_id, model_revision,
                        str(n_sample), frame_sampling_version, dtype, dimension_source])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: pathlib.Path, model_name: str, key: str) -> pathlib.Path:
    return pathlib.Path(cache_dir) / f"{model_name}_{key}.ndjson"


def load_cached(path: pathlib.Path) -> dict:
    """video_id -> embedding (list) onceden tamamlanmislardan. Dosya yoksa
    bos sozluk. Yarim kalmis/bozuk SON satir (crash tam yazma sirasinda
    olduysa) sessizce atlanir - digerlerini KAYBETMEZ, hepsini iptal etmez."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[row["video_id"]] = row["embedding"]
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def append_cached(path: pathlib.Path, video_id: str, embedding: list) -> None:
    """Her video TAMAMLANDIKCA cagrilir (toplu degil) - crash sonrasi
    ilerleme kaybolmasin diye."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"video_id": video_id, "embedding": list(embedding)},
                           ensure_ascii=False) + "\n")


__all__ = ["cache_key", "cache_path", "load_cached", "append_cached"]
