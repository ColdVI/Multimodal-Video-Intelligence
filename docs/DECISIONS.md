# Faz 7 kararları

- 2026-07-29 — Boş embedding notebook'u silinmeyip `notebooks/_archive/` altına alınacak; üretim kodu `service/` altında tutulacak.
- 2026-07-29 — pgvector 2048d HNSW sınırı nedeniyle `halfvec(2048)`; fp16 etkisini ayırmak için 1024d hem `vector` hem `halfvec` tutulacak.
- 2026-07-29 — `cached` modda sentetik query kullanılmayacak; yalnız `{dataset}_queries.npz` içindeki gerçek Qwen sorguları kabul edilecek, serbest yeni sorgu için `real` mod gerekecek.
- 2026-07-29 — Video oynatma kapsam dışı; UI kaynak yolu ve zaman aralığı gösterir.
