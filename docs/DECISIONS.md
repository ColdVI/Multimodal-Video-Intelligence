# Faz 7 kararları

## Faz 8 decisions

- 2026-07-30 - CapERA quality scope is test only: 1391 videos, 5 captions
  per video, exactly 6955 query/GT rows. Caption index is not treated as
  human/automatic provenance; caption_source remains unknown.
- 2026-07-30 - A0/system and A1/quality are separate readiness profiles.
  Missing A1 never blocks T1-T7.
- 2026-07-30 - Because pgvector 2048d is halfvec, exact cross-backend
  equality is gated only at float32-compatible 1024/512/256 dimensions.
  The 2048d result is a separate quantization experiment.
- 2026-07-30 - A/B/C values are labels, not distinct execution paths.
  T4 skips with reason pattern not implemented until real paths exist.
- 2026-07-30 - Negation and nonsense queries are exploratory rather than
  pass/fail gates. Paired bootstrap resamples video clusters, not query rows.

- 2026-07-30 — GPU/model indirmesi kritik yolu bloke etmesin diye standart servis imajı `synthetic/cached`, ayrı `docker-compose.gpu.yml` ise `real` Qwen modu için ayrıldı.
- 2026-07-30 — Embedding üretmeyen eski notebook 02 silinmedi; kanıt ve geçmiş korunarak `notebooks/_archive/` altına taşındı.
- 2026-07-30 — pgvector HNSW'nin 2000 boyut sınırı nedeniyle 2048d `halfvec`; fp16 cezasını ölçmek için 1024d hem `vector` hem `halfvec` tutulur.
- 2026-07-30 — Dataset görüntüsü indirmeden hazır, doğrulanmış 1.866 AU-AIR segmenti minimum uçtan uca veri yolu seçildi; SeaDronesSee ve MONET kritik yol dışında bırakıldı.
- 2026-07-30 — `cached` mod serbest metin için yalnız önceden üretilmiş `query_embeddings.json` girdilerini kabul eder; gerçek model yüklemeden bilinmeyen sorguya sahte vektör üretmez.
- 2026-07-30 — Kullanıcının açık talimatıyla önceki 7 commit ve Faz 7 çalışması doğrudan `main` üzerinde pushlanır; gece talimatındaki “push atma” kuralı bu teslim için geçersizdir.
- 2026-07-30 — Gradio'da yerleşik çift-tutamaklı range slider bulunmadığı gerçek konteyner importunda doğrulandı; her telemetri alanı aynı min/max semantiğini koruyan yan yana iki slider ile gösterilir.
- 2026-07-30 — Tam L2 matrisi uzun koşum olarak runner'da korunur; teslim artifact'i tüm 150 konfigürasyonu birer sorguyla ölçen ve `settings_json.execution=smoke` diye açık etiketlenen kısa koşumdur.
