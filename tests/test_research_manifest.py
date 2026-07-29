import json

from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest


def test_detect_hardware_profile_reports_local_cpu_when_no_cuda():
    # bu depodaki mevcut torch kurulumu CPU-only (bkz. BENCHMARK_CPU_GT1030_T4.md)
    result = detect_hardware_profile()
    assert result["hardware_profile"] == "local-cpu" or result["cuda_available"] is True
    assert "torch_version" in result


def test_write_manifest_includes_hardware_profile(tmp_path):
    m = RunManifest(notebook="00_test", hardware_profile="local-cpu", dataset_id="x")
    out_path = write_manifest(m, tmp_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["hardware_profile"] == "local-cpu"
    assert data["dataset_id"] == "x"
    assert "generated_at" in data


def test_manifest_requires_hardware_profile_field():
    m = RunManifest(notebook="x", hardware_profile="colab-T4-fp16-sdpa")
    assert m.hardware_profile == "colab-T4-fp16-sdpa"
