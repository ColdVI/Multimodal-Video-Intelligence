-- Model başına ayrı tablo (bilinçli POC basitleştirmesi - bkz. CONTEXT.md
-- "POC'ta bilinçli basitleştirmeler"). Üretimde raporun tek-tablo/çift-kolon
-- deseni doğru kalır; burada 2+ farklı boyutlu modeli (512d, 1152d) aynı
-- HNSW indeksine sıkıştırmamak için ayrıştırıldı.

CREATE TABLE IF NOT EXISTS clips_xclip_hf_zeroshot (
    video_id      LowCardinality(String),
    t_start       Float32,
    t_end         Float32,
    person_count  UInt16,
    car_count     UInt16,
    bus_count     UInt16,
    truck_count   UInt16,
    is_night      Bool,
    camera_motion Float32,
    brightness    Float32,
    platform      LowCardinality(String) DEFAULT 'visdrone',
    embedding     Array(Float32),
    INDEX idx_vec embedding TYPE vector_similarity('hnsw', 'cosineDistance', 512)
        GRANULARITY 100000000,
    INDEX idx_bus bus_count TYPE minmax GRANULARITY 4,
    INDEX idx_person person_count TYPE minmax GRANULARITY 4
) ENGINE = MergeTree ORDER BY (video_id, t_start);

CREATE TABLE IF NOT EXISTS clips_siglip2_frameavg (
    video_id      LowCardinality(String),
    t_start       Float32,
    t_end         Float32,
    person_count  UInt16,
    car_count     UInt16,
    bus_count     UInt16,
    truck_count   UInt16,
    is_night      Bool,
    camera_motion Float32,
    brightness    Float32,
    platform      LowCardinality(String) DEFAULT 'visdrone',
    embedding     Array(Float32),
    INDEX idx_vec embedding TYPE vector_similarity('hnsw', 'cosineDistance', 1152)
        GRANULARITY 100000000,
    INDEX idx_bus bus_count TYPE minmax GRANULARITY 4,
    INDEX idx_person person_count TYPE minmax GRANULARITY 4
) ENGINE = MergeTree ORDER BY (video_id, t_start);

-- Yeni model eklerken: yukarıdaki bloğu kopyala, tablo adını
-- clips_<models/__init__.py'deki isim> yap, vector_similarity boyutunu
-- o modelin `dim` alanıyla eşle.
