# Telemetry Mapping

Bu dosya, dataset manifestindeki canonical telemetri alanlarını özetler.

Kanonik alanlar ve sözleşme ayrıntıları için [DATASET_MANIFEST.md](DATASET_MANIFEST.md) ve [DATASET_ONBOARDING_GUIDE.md](DATASET_ONBOARDING_GUIDE.md) dosyalarını kullanın.

Özet:

- `person_count`, `vehicle_count`, `bus_count` gibi alanlar filtre kolonlarıdır.
- Gerçek kurum telemetrisi geldiğinde bu eşleme dış kaynaktan beslenir.
- Buradaki yapı, ingest ve ClickHouse şemasındaki alanlarla uyumlu tutulur.