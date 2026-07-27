import importlib.util
import json
from collections import Counter
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "inspect_dataset.py"
SPEC = importlib.util.spec_from_file_location("inspect_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_classify_structure_detects_video_only():
    assert MODULE.classify_structure(Counter({".mp4": 5})) == "video dosyalari"


def test_classify_structure_detects_frames_only():
    assert MODULE.classify_structure(Counter({".jpg": 500})) == "kare dizinleri (video yok)"


def test_classify_structure_detects_both():
    result = MODULE.classify_structure(Counter({".mp4": 5, ".jpg": 500}))
    assert "ikisi de var" in result


def test_classify_structure_unknown_when_neither():
    assert "bilinmiyor" in MODULE.classify_structure(Counter({".txt": 3}))


def test_sample_paths_keeps_first_n_and_samples_rest():
    paths = [Path(f"f{i:03d}.txt") for i in range(300)]
    sampled = MODULE.sample_paths(paths, first_n=100, random_n=50, seed=0)
    assert len(sampled) == 150
    assert set(paths[:100]) <= set(sampled)  # ilk 100 hep dahil


def test_sample_paths_handles_fewer_files_than_first_n():
    paths = [Path(f"f{i}.txt") for i in range(10)]
    sampled = MODULE.sample_paths(paths, first_n=100, random_n=50)
    assert sampled == sorted(paths)


def test_guess_annotation_format_mot_csv(tmp_path):
    p = tmp_path / "ann.txt"
    p.write_text("1,1,10,20,30,40,1,1,0\n2,1,12,22,30,40,1,1,0\n", encoding="utf-8")
    assert MODULE.guess_annotation_format(p).startswith("MOT-csv")


def test_guess_annotation_format_yolo_txt(tmp_path):
    p = tmp_path / "ann.txt"
    p.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.05 0.05\n", encoding="utf-8")
    assert MODULE.guess_annotation_format(p).startswith("YOLO-txt")


def test_guess_annotation_format_coco_json(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"images": [], "annotations": [], "categories": []}), encoding="utf-8")
    assert MODULE.guess_annotation_format(p) == "COCO-json"


def test_guess_annotation_format_caption_json_list(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps([{"video_id": "v1", "caption": "a bus"}]), encoding="utf-8")
    assert "caption" in MODULE.guess_annotation_format(p)


def test_guess_annotation_format_caption_json_dict(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"v1": ["caption one", "caption two"]}), encoding="utf-8")
    assert "caption" in MODULE.guess_annotation_format(p)


def test_guess_annotation_format_pascal_voc_xml(tmp_path):
    p = tmp_path / "ann.xml"
    p.write_text("<annotation><object><name>bus</name></object></annotation>", encoding="utf-8")
    assert MODULE.guess_annotation_format(p) == "PascalVOC-xml"


def test_guess_annotation_format_unreadable_file_does_not_raise(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text("{not valid json", encoding="utf-8")
    result = MODULE.guess_annotation_format(p)
    assert "okunamadi" in result


def test_check_name_mismatch_finds_missing_annotations():
    videos = ["seq_a", "seq_b", "seq_c"]
    anns = ["seq_a", "seq_b"]
    assert MODULE.check_name_mismatch(videos, anns) == ["seq_c"]


def test_check_name_mismatch_empty_when_all_match():
    assert MODULE.check_name_mismatch(["a", "b"], ["a", "b"]) == []


def test_find_frame_directories_finds_leaf_dirs_with_images(tmp_path):
    (tmp_path / "sequences" / "seq1").mkdir(parents=True)
    (tmp_path / "sequences" / "seq2").mkdir(parents=True)
    (tmp_path / "sequences" / "seq1" / "0000001.jpg").write_bytes(b"x")
    (tmp_path / "sequences" / "seq2" / "0000001.jpg").write_bytes(b"x")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "seq1.txt").write_text("1,1,1,1,1,1,1,1,1", encoding="utf-8")

    dirs = MODULE.find_frame_directories(tmp_path)
    names = {d.name for d in dirs}
    assert names == {"seq1", "seq2"}
    assert "annotations" not in names  # sadece resim iceren yapraklar, ust klasorler degil


def test_red_flags_detects_missing_fps_and_license():
    report = {"fps_manifest_found": False, "video_samples": [], "license_file_found": None, "name_mismatch": []}
    flags = MODULE.red_flags(report)
    assert any("fps" in f for f in flags)
    assert any("LICENSE" in f for f in flags)


def test_red_flags_clean_report_has_no_flags():
    report = {
        "fps_manifest_found": True, "video_samples": [],
        "license_file_found": "LICENSE", "name_mismatch": [],
    }
    assert MODULE.red_flags(report) == []


def test_red_flags_detects_wide_keyframe_interval_and_odd_height():
    report = {
        "fps_manifest_found": True,
        "video_samples": [{
            "path": "v.mp4", "height": 1071, "odd_height": True,
            "keyframe_estimate": {"estimated_keyframe_interval_s": 8.0},
        }],
        "license_file_found": "LICENSE", "name_mismatch": [],
    }
    flags = MODULE.red_flags(report)
    assert any("keyframe" in f for f in flags)
    assert any("yukseklik tek sayi" in f for f in flags)


def test_generate_adapter_skeleton_is_valid_python(tmp_path):
    report = {
        "structure": "video dosyalari", "annotation_format_guess": "MOT-csv",
        "fps_manifest_found": False, "license_file_found": None,
        "caption_detection": {"has_captions": False},
    }
    source = MODULE.generate_adapter_skeleton("my_dataset", report)
    assert "class MyDatasetAdapter(DatasetAdapter):" in source
    compile(source, "<generated>", "exec")  # sozdizimi gecerli mi
