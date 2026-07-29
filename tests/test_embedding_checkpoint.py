import pathlib

from src.research.embedding_checkpoint import (CheckpointWriter, already_done,
                                                remaining_items)


def test_buffer_flushes_at_flush_every(tmp_path):
    path = tmp_path / "ckpt.ndjson"
    w = CheckpointWriter(path, flush_every=3)
    w.add("a", [1.0, 2.0])
    w.add("b", [3.0, 4.0])
    assert not path.exists()  # henuz flush edilmedi
    w.add("c", [5.0, 6.0])
    assert path.exists()  # 3'e ulasti, flush oldu
    assert w.n_flushed == 3


def test_close_flushes_partial_buffer(tmp_path):
    path = tmp_path / "ckpt.ndjson"
    w = CheckpointWriter(path, flush_every=100)
    w.add("a", [1.0])
    w.add("b", [2.0])
    w.close()
    assert already_done(path) == {"a", "b"}


def test_resume_skips_already_done_items(tmp_path):
    path = tmp_path / "ckpt.ndjson"
    w = CheckpointWriter(path, flush_every=1)
    w.add("a", [1.0])
    w.add("b", [2.0])
    all_ids = ["a", "b", "c", "d"]
    remaining = remaining_items(all_ids, path)
    assert remaining == ["c", "d"]


def test_context_manager_flushes_on_exit(tmp_path):
    path = tmp_path / "ckpt.ndjson"
    with CheckpointWriter(path, flush_every=1000) as w:
        w.add("x", [1.0, 2.0, 3.0])
    assert already_done(path) == {"x"}


def test_already_done_empty_when_no_checkpoint_file(tmp_path):
    path = tmp_path / "does_not_exist.ndjson"
    assert already_done(path) == set()
    assert remaining_items(["a", "b"], path) == ["a", "b"]


def test_crash_mid_buffer_loses_at_most_flush_every_minus_one(tmp_path):
    """gercek crash simulasyonu: flush() hic cagrilmadan biterse tampondaki
    item'lar diske YAZILMAMIS olur - bu flush_every-1 item'lik beklenen
    maksimum kayip, sinirsiz kayip DEGIL."""
    path = tmp_path / "ckpt.ndjson"
    w = CheckpointWriter(path, flush_every=5)
    for i in range(4):
        w.add(f"item{i}", [float(i)])
    # close() / flush() KASITLI cagrilmadi - crash simulasyonu
    assert already_done(path) == set()  # henuz hic flush olmadi
    # ama bir sonraki kosuda bu 4 item TEKRAR islenecek (kabul edilebilir,
    # tekrar embed etmek pahali degil kadar - flush_every kucuk tutulmali)
