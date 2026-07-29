"""CapERAAdapter'i GERCEK yerel caption JSON'larina karsi test eder. Video
dosyalari bu depoda yok (bkz. datasets/capera.py docstring'i) - o kismi
test etmiyoruz, video-bagimsiz sozlesmeyi test ediyoruz."""
import pathlib

import pytest

from datasets.capera import CapERAAdapter, FIXED_DURATION_S

_HAS_CAPERA = (pathlib.Path("data/downloads/capera/CapERA_DATASET_test.json").exists()
              and pathlib.Path("data/downloads/capera/CapERA_DATASET_train.json").exists())
pytestmark = pytest.mark.skipif(not _HAS_CAPERA, reason="CapERA caption JSON'lari yok")


def test_list_sequences_matches_real_split_qualified_count():
    ce = CapERAAdapter()
    seqs = ce.list_sequences()
    # gercek: 1391 test + 1473 train = 2864 (split-nitelenmis, cakisma OLMAZ)
    assert len(seqs) == 2864
    assert "test__Baseball_001.mp4" in seqs
    assert "train__Baseball_001.mp4" in seqs  # AYNI dosya adi, FARKLI split - cakismamali


def test_split_qualification_prevents_id_collision():
    ce = CapERAAdapter()
    seqs = set(ce.list_sequences())
    # notebook'un kendi tespiti: ayni video_id iki splitte de var olabiliyor
    raw_ids = [s.split("__", 1)[1] for s in seqs]
    assert len(raw_ids) != len(set(raw_ids))  # ham video_id'lerde cakisma VAR
    assert len(seqs) == len(set(seqs))  # ama split-nitelenmis kimliklerde YOK


def test_ground_truth_returns_five_captions_all_pointing_to_same_interval():
    ce = CapERAAdapter()
    gt = ce.ground_truth("test__Baseball_001.mp4")
    assert len(gt) == 5
    assert all(intervals == [(0.0, FIXED_DURATION_S)] for intervals in gt.values())


def test_ground_truth_unknown_sequence_raises_keyerror():
    ce = CapERAAdapter()
    with pytest.raises(KeyError):
        ce.ground_truth("does_not_exist.mp4")


def test_captions_returns_five_items():
    ce = CapERAAdapter()
    caps = ce.captions("test__Baseball_001.mp4")
    assert len(caps) == 5
    assert all(isinstance(c, str) and c for c in caps)


def test_load_video_constructs_split_and_category_qualified_path():
    ce = CapERAAdapter()
    path = ce.load_video("test__Baseball_001.mp4")
    assert path.parts[-3:] == ("test", "Baseball", "Baseball_001.mp4")


def test_fps_raises_not_implemented_rather_than_guessing():
    ce = CapERAAdapter()
    with pytest.raises(NotImplementedError):
        ce.fps("test__Baseball_001.mp4")


def test_license_is_explicitly_unknown_not_guessed():
    assert "UNKNOWN" in CapERAAdapter().license()


def test_manifest_query_count_matches_5x_item_count():
    ce = CapERAAdapter()
    m = ce.manifest()
    assert m.item_count == 2864
    assert m.query_count == 2864 * 5
    assert m.dataset_id == "capera"
    assert m.has_structured_filters is False
    assert m.retrieval_unit == "video"


def test_manifest_source_hash_matches_real_concatenated_file_hash():
    import hashlib
    ce = CapERAAdapter()
    paths = ce._paths()
    expected = hashlib.sha256(paths["test"].read_bytes() + paths["train"].read_bytes()).hexdigest()
    assert ce.manifest().source_hash == expected


def test_get_dataset_adapter_registers_capera():
    from datasets import available_datasets, get_dataset_adapter
    assert "capera" in available_datasets()
    ce = get_dataset_adapter("capera")
    assert isinstance(ce, CapERAAdapter)
