"""VisDroneAdapter/MSRVTTAdapter'i GERCEK yerel veriye karsi test eder
(bu repoda kurulu kalici pratik - bkz. tests/test_inspect_dataset.py,
tests/test_validate_msrvtt.py). Veri yoksa (temiz checkout) ilgili test
sinifi atlanir."""
import pathlib

import pytest

from dataset_adapters import get_dataset_adapter, qualified_id
from dataset_adapters.msrvtt import MSRVTTAdapter
from dataset_adapters.visdrone import VisDroneAdapter

_HAS_VISDRONE = pathlib.Path("data/raw/manifest.json").exists()
_HAS_MSRVTT = pathlib.Path("data/downloads/msrvtt/msrvtt_test_1k.json").exists()


@pytest.mark.skipif(not _HAS_VISDRONE, reason="data/raw/manifest.json yok (VisDrone indirilmemis)")
class TestVisDroneAdapter:
    def test_list_sequences_matches_real_manifest(self):
        vd = VisDroneAdapter()
        seqs = vd.list_sequences()
        assert len(seqs) == 56
        assert "uav0000013_01073_v" in seqs

    def test_load_video_points_to_existing_file(self):
        vd = VisDroneAdapter()
        path = vd.load_video(vd.list_sequences()[0])
        assert path.exists()
        assert path.suffix == ".mp4"

    def test_fps_is_real_number_from_manifest(self):
        vd = VisDroneAdapter()
        assert vd.fps("uav0000013_01073_v") == 25

    def test_ground_truth_matches_known_real_annotation_result(self):
        vd = VisDroneAdapter()
        gt = vd.ground_truth("uav0000013_01073_v")
        assert "yayaları göster" in gt
        assert "show the pedestrians" in gt
        assert gt["yayaları göster"] == gt["show the pedestrians"]  # TR/EN ayni GT fn'i paylasir

    def test_ground_truth_unknown_sequence_returns_empty_dict(self):
        vd = VisDroneAdapter()
        assert vd.ground_truth("uav_does_not_exist") == {}

    def test_manifest_query_count_is_28_not_6(self):
        vd = VisDroneAdapter()
        m = vd.manifest()
        assert m.query_count == 28
        assert m.item_count == 56
        assert m.dataset_id == "visdrone"
        assert m.has_structured_filters is True

    def test_window_params_match_config(self):
        vd = VisDroneAdapter()
        wp = vd.window_params()
        assert wp == {"size_s": 8.0, "stride_s": 4.0, "min_s": 2.0}

    def test_license_is_not_empty_or_none(self):
        assert VisDroneAdapter().license()


@pytest.mark.skipif(not _HAS_MSRVTT, reason="MSR-VTT split dosyasi yok (indirilmemis)")
class TestMSRVTTAdapter:
    def test_list_sequences_has_1000_entries(self):
        mv = MSRVTTAdapter()
        assert len(mv.list_sequences()) == 1000

    def test_load_video_points_to_existing_file(self):
        mv = MSRVTTAdapter()
        path = mv.load_video(mv.list_sequences()[0])
        assert path.exists()

    def test_captions_returns_single_item_list(self):
        mv = MSRVTTAdapter()
        seq = mv.list_sequences()[0]
        caps = mv.captions(seq)
        assert isinstance(caps, list)
        assert len(caps) == 1

    def test_ground_truth_whole_clip_is_one_interval_from_zero(self):
        mv = MSRVTTAdapter()
        seq = mv.list_sequences()[0]
        gt = mv.ground_truth(seq)
        assert len(gt) == 1
        (caption, intervals), = gt.items()
        assert caption == mv.captions(seq)[0]
        assert len(intervals) == 1
        t0, t1 = intervals[0]
        assert t0 == 0.0
        assert t1 > 0.0

    def test_manifest_query_count_is_1000_not_28(self):
        mv = MSRVTTAdapter()
        m = mv.manifest()
        assert m.query_count == 1000
        assert m.item_count == 1000
        assert m.dataset_id == "msrvtt_1ka"
        assert m.has_structured_filters is False
        assert m.retrieval_unit == "video"

    def test_manifest_source_hash_matches_real_file_hash(self):
        import hashlib
        mv = MSRVTTAdapter()
        expected = hashlib.sha256(mv._split_path().read_bytes()).hexdigest()
        assert mv.manifest().source_hash == expected

    def test_license_is_explicitly_unknown_not_guessed(self):
        # bu oturumda MSR-VTT lisansi dogrulanamadi - UNKNOWN'i UYDURULMUS
        # bir SPDX etiketiyle GIZLEMEK yerine acikca yazmak dogru davranis.
        assert "UNKNOWN" in MSRVTTAdapter().license()


def test_qualified_id_prevents_cross_dataset_collision_even_with_same_video_id():
    a = qualified_id("visdrone", "video7020", 0.0)
    b = qualified_id("msrvtt_1ka", "video7020", None)
    assert a != b
    assert a[0] != b[0]


@pytest.mark.skipif(not (_HAS_VISDRONE and _HAS_MSRVTT), reason="her iki dataset da gerekli")
def test_get_dataset_adapter_returns_cached_singleton_per_dataset_id():
    a1 = get_dataset_adapter("visdrone")
    a2 = get_dataset_adapter("visdrone")
    assert a1 is a2
    assert isinstance(a1, VisDroneAdapter)
