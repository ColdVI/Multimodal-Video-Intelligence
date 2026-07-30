# Faz 7 kararları

- 2026-07-30 — GPU/model indirmesi kritik yolu bloke etmesin diye standart servis imajı `synthetic/cached`, ayrı `docker-compose.gpu.yml` ise `real` Qwen modu için ayrıldı.
- 2026-07-30 — Embedding üretmeyen eski notebook 02 silinmedi; kanıt ve geçmiş korunarak `notebooks/_archive/` altına taşındı.
- 2026-07-30 — pgvector HNSW'nin 2000 boyut sınırı nedeniyle 2048d `halfvec`; fp16 cezasını ölçmek için 1024d hem `vector` hem `halfvec` tutulur.
- 2026-07-30 — Dataset görüntüsü indirmeden hazır, doğrulanmış 1.866 AU-AIR segmenti minimum uçtan uca veri yolu seçildi; SeaDronesSee ve MONET kritik yol dışında bırakıldı.
- 2026-07-30 — `cached` mod serbest metin için yalnız önceden üretilmiş `query_embeddings.json` girdilerini kabul eder; gerçek model yüklemeden bilinmeyen sorguya sahte vektör üretmez.
- 2026-07-30 — Kullanıcının açık talimatıyla önceki 7 commit ve Faz 7 çalışması doğrudan `main` üzerinde pushlanır; gece talimatındaki “push atma” kuralı bu teslim için geçersizdir.

