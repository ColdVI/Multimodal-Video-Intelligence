# Faz 7 blokerleri

- 2026-07-29 — Denenen: `docker compose -f docker-compose.faz7.yml up -d --build pg ch qdrant api`; sonuç: Docker socket'e bağlanılamadı (`Is the docker daemon running?`); neden devam edilemedi: host Docker daemon kapalı; bağımsız sıradaki iş: saf mantık testleri, API/UI/benchmark/notebook ve doğrulama betiği tamamlanıyor.
- 2026-07-29 — SeaDronesSee/MONET indirmesi denenmedi; kritik yol gerçek ve hazır AU-AIR parquet'leriyle sınırlandı; sıradaki adım: kullanıcı isterse resmi annotation JSON'unu `data/downloads/` altına koyup loader'ı çalıştırmak.
