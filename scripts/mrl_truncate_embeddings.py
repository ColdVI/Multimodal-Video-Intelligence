"""Faz 4 MRL boyut taramasi: 2048d Qwen3-VL-Embedding ciktisini yeniden
modeli kosturmadan 1024/512/256'ya kirpar + L2-yeniden-normalize eder (MRL'in
kendi tanimi budur - ilk N boyut anlamli bir alt-temsil tasir). Boylece 4
boyutun karsilastirmasi tek gercek embed_video() kosumundan cikar."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

DIMS = (1024, 512, 256)


def truncate_and_renormalize(vec: list, dim: int) -> list:
    arr = np.asarray(vec[:dim], dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def main():
    src_path = pathlib.Path("data/embeddings_qwen3vl_emb_2048.json")
    if not src_path.exists():
        print(f"HATA: {src_path} yok. Once: python ingest/03_embed.py --model qwen3vl_emb_2048")
        raise SystemExit(1)

    entries = json.load(open(src_path, encoding="utf-8"))
    for dim in DIMS:
        out = []
        for e in entries:
            out.append({**e, "embedding": truncate_and_renormalize(e["embedding"], dim)})
        out_path = pathlib.Path(f"data/embeddings_qwen3vl_emb_{dim}.json")
        json.dump(out, open(out_path, "w", encoding="utf-8"))
        print(f"{len(out)} embedding ({dim}d, 2048d'den kirpildi) -> {out_path}")


if __name__ == "__main__":
    main()
