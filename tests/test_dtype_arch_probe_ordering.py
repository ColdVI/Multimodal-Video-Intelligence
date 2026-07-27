"""Gercek bir Colab kosumunda yakalanan bug: nn.Module.half() YERINDE
(in-place) mutasyon yapar ve self dondurur - model_fp16 ve model AYNI
nesne. torch.compile testi fp16 donusumunden SONRA calisirsa aslinda
"fp16 + compile" olcer, "native dtype + compile" degil. Bu test compile
testinin fp16 mutasyonundan ONCE calistigini kilitler."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")


class _FakeModel:
    """SentenceTransformer'in probe_model()'in kullandigi yuzeyini taklit
    eder: encode(), parameters(), half() (yerinde mutasyon + self donusu,
    gercek nn.Module davranisiyla ayni)."""

    def __init__(self):
        self.dtype_state = "native"
        self.calls = []  # her encode() cagrisinda dtype_state kaydedilir
        self._modules = {}

    def parameters(self):
        yield torch.zeros(1)

    def encode(self, *args, **kwargs):
        self.calls.append(self.dtype_state)
        return [[0.0]]

    def half(self):
        self.dtype_state = "fp16"
        return self  # gercek nn.Module.half() davranisi


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    fake_model = _FakeModel()
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda *a, **kw: fake_model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(torch, "compile", lambda m: m, raising=True)
    # Bu gelistirme makinesinde gercek CUDA yok - fp16 dalini (probe_model
    # icinde "if device == 'cuda':" ile korunuyor) yine de test edebilmek
    # icin is_available/synchronize'i sahteliyoruz (gercek CUDA cagrisi
    # olmadan calissin).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=True)
    return fake_model


def test_compile_runs_before_fp16_mutation_not_after(fake_sentence_transformers, monkeypatch):
    from scripts.dtype_arch_probe import probe_model

    probe_model("fake/model", image_size=4, skip_compile=False)

    calls = fake_sentence_transformers.calls
    # sira: native_dtype_timing (n=10, warmup=3 -> 13 cagri, hepsi "native"),
    # sonra torch_compile_timing (n=5, warmup=1 -> 6 cagri, HALA "native"
    # olmali - asil regresyon kontrolu budur), sonra fp16_timing (13 cagri,
    # hepsi "fp16").
    native_and_compile_calls = calls[:13 + 6]
    assert all(c == "native" for c in native_and_compile_calls), (
        "torch.compile testi fp16 mutasyonundan SONRA calismis - "
        "aslinda 'fp16 + compile' olcuyor, 'native dtype + compile' degil")
    assert calls[-1] == "fp16"
