"""Her bench run'i icin insan/test okunur run manifest'i (Faz 1 madde 2):
git hash, OS/Python/Torch/ClickHouse surumleri, GPU adi+VRAM, model ID +
checkpoint revision + embedding boyutu + normalize durumu, config snapshot,
veri kapsami, toplam sure. Ikincil metadata (torch/clickhouse introspection)
toplanamazsa run'i dusurmez - hata mesaji alana yazilir."""
import copy
import platform
import subprocess
import time


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unavailable"
    except Exception as exc:
        return f"unavailable: {exc}"


def torch_info() -> dict:
    try:
        import torch
        info = {"version": torch.__version__, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            info["gpu_name"] = props.name
            info["gpu_vram_mb"] = round(props.total_memory / (1024 * 1024))
        return info
    except Exception as exc:
        return {"error": str(exc)}


def clickhouse_version(cfg: dict) -> str:
    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=cfg["clickhouse"]["host"], port=cfg["clickhouse"]["port"])
        return client.query("SELECT version()").result_rows[0][0]
    except Exception as exc:
        return f"unavailable: {exc}"


def model_info(model_name: str) -> dict:
    try:
        from models import get_embedder
        emb = get_embedder(model_name)
        return {"name": emb.name, "dim": emb.dim, "l2_normalized": True}
    except Exception as exc:
        return {"error": str(exc)}


def capture_run_manifest(spec, cfg: dict, data_scope: dict = None,
                          duration_s: float = None, extra: dict = None) -> dict:
    manifest = {
        "run_id": spec.run_id,
        "spec": spec.as_dict(),
        "git_hash": git_hash(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_info(),
        "clickhouse_version": clickhouse_version(cfg),
        "model": model_info(spec.model_name),
        "config_snapshot": copy.deepcopy(cfg),
        "data_scope": data_scope or {},
        "duration_s": duration_s,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        manifest.update(extra)
    return manifest
