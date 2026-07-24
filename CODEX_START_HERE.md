# Codex başlangıç noktası

Bu klasör, çalışan POC koduyla birlikte Codex'e verilmeye hazır bağlam ve
konuşma dosyalarını içerir. Arşivi açtıktan sonra Codex'i bu klasörde başlatın.

## En kısa kullanım

1. Yeni bir Codex konuşması açın.
2. Çalışma klasörü olarak bu repo kökünü seçin.
3. `docs/codex/00_TEK_SEFERLIK_ANA_PROMPT.md` içindeki metni yapıştırın.
4. Veri yoksa `scripts/poc.ps1 download-data` resmî Task 4 bağlantısından
   indirip boyut/SHA/veri sözleşmesini doğrular. Kota veya giriş engeli çıkarsa
   `docs/codex/01_ASAMALI_KONUSMALAR.md` içindeki veri adımını kullanın.

Uzun işi tek konuşmaya vermek istemiyorsanız ana prompt yerine
`01_ASAMALI_KONUSMALAR.md` dosyasındaki mesajları sırayla kullanın.

## Codex'in okuyacağı kaynakların öncelik sırası

Bir çelişki olursa şu sıra geçerlidir:

1. Çalışan kod ve testlerdeki fonksiyon sözleşmeleri.
2. `AGENTS.md` — doğrulanan/doğrulanmayan işler ve insan müdahalesi.
3. `CONTEXT.md` — mimari gerekçeler ve bilinçli basitleştirmeler.
4. `TASKS.md` — uygulanacak checklist.
5. `docs/codex/` — çalışma promptları, plan ve kabul kriterleri.
6. Eski `hibrit-video-arama-poc-plani.md` yalnızca tarihsel fikir kaynağıdır;
   içindeki tek embedding tablosu gibi eski örnekler doğrudan koda taşınmaz.

## Teslim anındaki doğrulama durumu

24 Temmuz 2026'da bu kopyada yeniden doğrulandı:

- `pytest tests/ -v`: **46/46 geçti**.
- 40 Python dosyası `python -m py_compile` ile geçti.
- Test ortamı: Windows, Python 3.14.6, pytest 9.1.1.
- Güncel Windows runner pytest cache eklentisini kapatır; son koşu uyarısız geçti.

Resmî VisDrone-MOT verisi ve 5 gerçek sekans üzerinde X-CLIP, SigLIP2, YOLO,
ClickHouse, GT ve filtre A/B smoke kapıları geçti; güncel tek doğru durum kaynağı
`STATUS.md` dosyasıdır. Görsel GT, büyük eval ve ölçek testi bekliyor.

Bu ayrım korunmalıdır: test edilmemiş bir adım hiçbir raporda “tamamlandı” diye
yazılmaz.

## Dosya haritası

- `docs/codex/00_TEK_SEFERLIK_ANA_PROMPT.md`: Tek mesajda tam görev.
- `docs/codex/01_ASAMALI_KONUSMALAR.md`: Küçük ve denetlenebilir Codex turları.
- `docs/codex/02_FIKIRLER_VE_KARARLAR.md`: Korunacak kararlar ve geliştirme fikirleri.
- `docs/codex/03_UYGULAMA_PLANI.md`: Fazlar, girişler, kanıtlar ve karar kapıları.
- `docs/codex/04_KABUL_KRITERLERI_VE_RAPOR.md`: Bitti tanımı ve rapor şablonu.
- `AGENTS.md`, `CONTEXT.md`, `TASKS.md`: Reponun mevcut teknik handoff'u.

## Veri kaynağı güvenlik kapısı

Yalnızca resmî VisDrone GitHub deposundaki Task 4 Google Drive kimliği ve
`scripts/download_visdrone.py` kullanılmalıdır. Script boyut, SHA-256, ZIP yol
güvenliği ve 56/56/24.201 sözleşmesini doğrular. Bağlantı giriş/kota isterse
Codex kimlik bilgisi istememeli ve kullanıcıdan resmî dosyayı istemelidir.
