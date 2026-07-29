import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.verify_drive_inputs import _check_folder


def test_check_folder_reports_missing_when_absent():
    result = _check_folder("definitely_does_not_exist_xyz", [])
    assert result["exists"] is False
    assert result["n_files_found"] == 0


def test_check_folder_root_dot_does_not_recurse_into_subfolders(tmp_path, monkeypatch):
    import src.research.colab_paths as colab_paths
    (tmp_path / "top_level.txt").write_text("x")
    sub = tmp_path / "datasets"
    sub.mkdir()
    (sub / "nested.txt").write_text("y")
    monkeypatch.setattr(colab_paths, "research_root", lambda: tmp_path)

    result = _check_folder(".", [])
    assert result["n_files_found"] == 1
    assert result["sample_found"] == ["top_level.txt"]


def test_drive_manifest_json_is_valid_and_has_required_keys():
    manifest = json.loads(pathlib.Path("drive_manifest.json").read_text(encoding="utf-8"))
    assert "drive_root" in manifest
    assert "folders" in manifest
    assert "datasets/auair" in manifest["folders"]
