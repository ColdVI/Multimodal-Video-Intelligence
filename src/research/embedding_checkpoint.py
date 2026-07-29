"""GPU asamasi embedding uretimi icin toplu (batched) checkpoint/resume
katmani. scripts/msrvtt_embedding_cache.py::load_cached/append_cached
TEK KAYNAK - burada YENIDEN YAZILMAZ, sadece Drive'a HER item'da degil
HER N item'da (varsayilan 100) yazacak sekilde TAMPONLANIR (Colab'da
Drive'a cok sayida kucuk yazma yavastir - kullanicinin acik istegi:
'her 100 veya 200 item'da checkpoint')."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.msrvtt_embedding_cache import append_cached, load_cached  # noqa: E402


class CheckpointWriter:
    """add() cagrildikca tamponlar, flush_every'ye ulasinca (veya close()
    cagrilinca) diske yazar. Crash EN FAZLA flush_every-1 item kaybettirir -
    hicbir zaman TUMUNU degil (append_cached zaten per-item, tamponun
    kendisi crash aninda kaybolabilecek TEK kisim)."""

    def __init__(self, path: pathlib.Path, flush_every: int = 100):
        self.path = pathlib.Path(path)
        self.flush_every = flush_every
        self._buffer = []
        self.n_flushed = 0

    def add(self, item_id: str, embedding) -> None:
        self._buffer.append((item_id, list(embedding)))
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> int:
        n = len(self._buffer)
        for item_id, emb in self._buffer:
            append_cached(self.path, item_id, emb)
        self.n_flushed += n
        self._buffer = []
        return n

    def close(self) -> int:
        return self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def already_done(path: pathlib.Path) -> set:
    """resume: bu checkpoint dosyasinda ZATEN tamamlanmis item_id kumesi."""
    return set(load_cached(path).keys())


def remaining_items(all_ids: list, path: pathlib.Path) -> list:
    done = already_done(path)
    return [i for i in all_ids if i not in done]


__all__ = ["CheckpointWriter", "already_done", "remaining_items", "load_cached", "append_cached"]
