# Notebook 05 - Hybrid sorgu benchmarki sonucu

## Faz 8 synthetic ANN interpretability correction

Synthetic i.i.d. Gaussian unit vectors have no cluster structure. Therefore
ANN recall values from ClickHouse, pgvector, and Qdrant in synthetic mode
cannot rank backend quality; Faz 8 rows carry interpretable=false. Valid
synthetic claims are system integrity, filter correctness, latency, and
float32 exact agreement with the stable NumPy reference. Backend quality
must wait for T8 with real CapERA test embeddings.

## Durum: KANIT_YOK

- can_run_real_benchmark=False
- healthy_backends (notebook 04'ten)=[]
- common_exact_ready=False
- uretilen satir sayisi=0

Bu ortamda (Colab disi/backend kurulamadi) benchmark KOSULMADI. vector_database_results.csv BOS yazildi - sifir/varsayilan deger UYDURULMADI, dosya kendisi 'veri yok' anlamina gelir.
