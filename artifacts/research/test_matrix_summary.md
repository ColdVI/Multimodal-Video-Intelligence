# Faz 8 test matrix summary

| Sınıf | Koşu | Geçti | Kaldı | Skip | Exploratory | Skip nedeni | Yorumlanabilir |
|---|---:|---:|---:|---:|---:|---|---|
| T1 | 13 | 12 | 0 | 0 | 1 | — | evet |
| T2 | 3 | 3 | 0 | 0 | 0 | — | evet |
| T3 | 288 | 288 | 0 | 0 | 0 | — | kısmen/hayır |
| T4 | 1 | 0 | 0 | 1 | 0 | pattern not implemented | evet |
| T5 | 4 | 4 | 0 | 0 | 0 | — | evet |
| T6 | 5 | 5 | 0 | 0 | 0 | — | evet |
| T7 | 5 | 5 | 0 | 0 | 0 | — | evet |
| T8 | 1 | 0 | 0 | 1 | 0 | A1.1: missing: capera_2048.npy, capera_ids.parquet, capera_queries_2048.npy, capera_query_ids.parquet, query_embeddings.json, embedding_manifest.json; A1.2: ValueError: CapERA cached ingest missing; A1.3: embedding_mode=synthetic, selected_mode=hybrid_text, warm_p50_ms=738.879, synthetic_fallback=False; A1.4: groundtruth={'rows': 0, 'unique_queries': 0, 'videos': 0, 'caption_source_unknown': 0, 'non_test_video_ids': 0}; A1.5: UI reachable; non-synthetic mode required | evet |
| T9 | 1 | 1 | 0 | 0 | 0 | — | evet |
| T10 | 1 | 0 | 0 | 1 | 0 | Playwright package/browser not installed | evet |
