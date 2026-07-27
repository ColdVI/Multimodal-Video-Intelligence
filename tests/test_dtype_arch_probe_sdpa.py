"""SDPA backend-ayristirma ve compile-matrisi genisletmesi icin testler.
Gercek Qwen3-VL-Embedding-2B'yi indirip GPU'da calistirmiyor (bu makinede
GPU yok, model agir) - sahte bir SentenceTransformer ile YALNIZCA
"kac kez model yuklendi / hangi hucreler dogru sirayla calisti" gibi
yapisal davranisi dogruluyor. Gercek zamanlama sayilari kullanicinin
T4/L4 oturumunu gerektirir."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from scripts.dtype_arch_probe import decompose_speedup, probe_compile_matrix, probe_sdpa_backend_matrix


class _FakeModel:
    def __init__(self, load_id):
        self.load_id = load_id  # her yukleme cagrisinda benzersiz - "temiz yeniden yukleme" kontrolu icin
        self.is_half = False

    def encode(self, *args, **kwargs):
        return [[0.0]]

    def half(self):
        self.is_half = True
        return self


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    created = []

    def factory(*a, **kw):
        m = _FakeModel(load_id=len(created))
        created.append(m)
        return m

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(torch, "compile", lambda m: m, raising=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=True)
    return created


def test_decompose_speedup_computes_attention_and_dtype_ratios():
    matrix = {
        "native_bf16/MATH": {"median_s": 10.0},
        "native_bf16/FLASH_ATTENTION": {"median_s": 2.0},
        "fp16/MATH": {"median_s": 1.0},
        "fp16/FLASH_ATTENTION": {"median_s": 0.5},
        "native_bf16/EFFICIENT_ATTENTION": {"error": "reddedildi"},
    }
    result = decompose_speedup(matrix)

    assert result["attention_backend_ratio_native_bf16"]["ratio_x"] == 5.0  # 10/2
    assert result["attention_backend_ratio_fp16"]["ratio_x"] == 2.0  # 1.0/0.5
    assert result["gemm_dtype_ratio_MATH"]["ratio_x"] == 10.0  # 10.0/1.0
    assert result["gemm_dtype_ratio_FLASH_ATTENTION"]["ratio_x"] == 4.0  # 2.0/0.5
    assert "gemm_dtype_ratio_EFFICIENT_ATTENTION" not in result  # bir taraf hata verdi


def test_decompose_speedup_empty_when_nothing_succeeded():
    assert decompose_speedup({"native_bf16/MATH": {"error": "x"}}) == {}


def test_probe_sdpa_backend_matrix_reloads_model_once_per_dtype_not_per_cell(fake_sentence_transformers):
    matrix = probe_sdpa_backend_matrix("fake/model", image_size=4)

    # 2 dtype x 3 backend = 6 hucre, ama SADECE 2 model yuklemesi olmali
    # (backend degisiminde model DEGISMIYOR, sdpa_kernel sadece dispatch'i
    # etkiliyor - yeniden yuklemeye gerek yok, sadece dtype degisince var)
    assert len(fake_sentence_transformers) == 2
    assert len(matrix) == 6
    assert all("median_s" in v for v in matrix.values())


def test_probe_sdpa_backend_matrix_second_load_is_half(fake_sentence_transformers):
    probe_sdpa_backend_matrix("fake/model", image_size=4)
    assert fake_sentence_transformers[0].is_half is False  # native_bf16
    assert fake_sentence_transformers[1].is_half is True   # fp16


def test_probe_sdpa_backend_matrix_skip_returns_skipped_marker():
    result = probe_sdpa_backend_matrix("fake/model", image_size=4, skip=True)
    assert result == {"skipped": "--skip-sdpa-matrix"}


def test_probe_compile_matrix_loads_four_fully_independent_models(fake_sentence_transformers):
    results = probe_compile_matrix("fake/model", image_size=4)

    # bf16, bf16_compiled, fp16, fp16_compiled - HER BIRI icin AYRI yukleme,
    # onceki hucrenin half()'lanmis modelini paylasmiyor (regresyon: bu tam
    # olarak daha once yakalanan bug'in genellemesi - .half() yerinde
    # mutasyon yapiyor, paylasilan nesne kullanmak sonraki olcumu kirletir)
    assert len(fake_sentence_transformers) == 4
    assert set(results.keys()) == {"native_bf16", "native_bf16_compiled", "fp16", "fp16_compiled"}

    # ilk iki yukleme (bf16, bf16_compiled) half() cagrilmamis olmali,
    # son iki yukleme (fp16, fp16_compiled) half() cagrilmis olmali
    assert fake_sentence_transformers[0].is_half is False
    assert fake_sentence_transformers[1].is_half is False
    assert fake_sentence_transformers[2].is_half is True
    assert fake_sentence_transformers[3].is_half is True


def test_probe_compile_matrix_skip_returns_skipped_marker():
    assert probe_compile_matrix("fake/model", image_size=4, skip=True) == {"skipped": "--skip-compile"}
