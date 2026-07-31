# Dataset onboarding kılavuzu

Bu kılavuz, yeni bir kurum video/telemetri koleksiyonunu FAZ 11 manifest
sözleşmesine dönüştürmek için alan-alan referanstır. İki tam örnek manifest
mevcuttur — ikisi de gerçek `service/app/ingestion/manifest.py` parser'ı ile
doğrulanmıştır:

- `datasets/example_uav.yaml` — absolute clock (`unix_ms` + container
  creation time anchor), filename-stem pairing, AGL/ground_speed, drop_partial.
- `datasets/example_institution.yaml` — relative clock (anchor gerekmez),
  `manifest_csv` pairing, MSL/air_speed, pad_last.

Sözleşme özeti: [DATASET_MANIFEST.md](DATASET_MANIFEST.md). Bu dosya onun
adım-adım onboarding anlatımıdır.

## Desteklenen video formatları

Streaming decoder önce PyAV, PyAV yoksa OpenCV kullanır
(`service/app/ingestion/video.py::probe_video`). Pratikte standart H.264/H.265
MP4 container'ları desteklenir; codec bilgisi `probe_video()` çıktısında
görünür ve preflight'ta raporlanır. Codec/container garantisi verilmez —
`video_probe` preflight kontrolü sizin gerçek dosyanız üzerinde
çalıştırılmalıdır.

## Video dosya isimlendirmesi

`source.video_id_from`:

```yaml
video_id_from: filename_stem          # flight-001.mp4 -> flight-001
# veya
video_id_from: "regex:^(?P<video_id>.+?)_part\\d+$"   # flight-001_part2.mp4 -> flight-001
```

Regex, adlandırılmış `video_id` grubu içermek zorundadır; içermezse
`derive_identifier()` açık hata verir.

## Telemetry dosyası eşleştirmesi

İki strateji:

```yaml
pairing:
  strategy: filename_stem       # video ve telemetri aynı stem'i paylaşır
  telemetry_glob: "telemetry/**/*.csv"
  telemetry_id_from: filename_stem
  manifest_csv: null
```

veya karmaşık/parçalı eşlemeler için:

```yaml
pairing:
  strategy: manifest_csv
  telemetry_glob: null
  telemetry_id_from: filename_stem
  manifest_csv: "pairing/institution_flights_v2.csv"   # DATA_ROOT'a göreli
```

`manifest_csv` şu kolonları taşır (bkz. `datasets/example_institution.yaml`
yorumu):

```text
video_id,video_path,telemetry_path,video_start_unix_s,offset_s
```

`video_path`/`telemetry_path` de `DATA_ROOT`'a görelidir; mutlak yol veya
`..` reddedilir. `video_start_unix_s`/`offset_s` boş bırakılabilir (relative
clock kullanıyorsanız anchor'a hiç gerek yoktur).

## Relative clock örneği

```yaml
time_alignment:
  video_clock: pts
  telemetry_clock: relative_s
  video_start_time_from: null
  offset_s: 0.0
  max_gap_s: 1.5
```

`aligned_s = telemetry_relative_s - offset_s`. Video wall-clock anchor hiç
aranmaz; preflight bunu eksiklik saymaz. Tam örnek:
`datasets/example_institution.yaml`.

## Absolute Unix timestamp örneği

```yaml
time_alignment:
  video_clock: pts
  telemetry_clock: unix_ms   # veya unix_s
  video_start_time_from: container_creation_time
  offset_s: 0.0
```

`aligned_s = telemetry_unix_s - video_start_unix_s - offset_s`. Anchor
(`video_start_unix_s`) çözülemezse preflight FAIL olur — sahte anchor
üretilmez.

## Filename timestamp anchor örneği

```yaml
time_alignment:
  video_clock: pts
  telemetry_clock: unix_s
  video_start_time_from: filename
  filename_time_regex: '(?P<timestamp>\d{8}_\d{6})'
  filename_time_format: '%Y%m%d_%H%M%S'
  timezone: Europe/Istanbul
```

`flight_20260115_143000.mp4` gibi bir dosya adından anchor çıkarır. Regex
adlandırılmış `timestamp` grubu içermeli, `filename_time_format` bunu
`datetime.strptime` ile ayrıştırmalıdır. `timezone` burada hem bu anchor'ı
hem telemetri CSV'sindeki timezone-naive `iso8601` zaman damgalarını
yerelleştirmek için kullanılır (bkz. [DATASET_MANIFEST.md](DATASET_MANIFEST.md)'nin
timezone bölümü) — asla sessizce UTC varsayılmaz.

## Container creation time anchor örneği

```yaml
time_alignment:
  video_clock: pts
  telemetry_clock: unix_ms
  video_start_time_from: container_creation_time
```

MP4 container metadata'sındaki `creation_time`'ı kullanır
(`probe_video().creation_time`). Bu alan videoda yoksa preflight FAIL olur.
Tam örnek: `datasets/example_uav.yaml`.

## Offset formülü

```text
absolute: aligned_s = telemetry_unix_s   - video_start_unix_s - offset_s
relative: aligned_s = telemetry_relative_s                    - offset_s
```

Pozitif `offset_s`, telemetriyi video zaman çizelgesinde **daha erken**
konuma taşır (kod ve doküman aynı işareti kullanır).

## Canonical telemetri alanları

```text
event_category, split, video_id, latitude, longitude, altitude_m,
velocity_mps, roll, pitch, yaw, yaw_rate, gimbal_pitch, gimbal_heading,
compass_heading, person_count, vehicle_count, bus_count, is_night
```

Her biri `telemetry.fields` altında yalnız operatör açıkça bir `source`
kolonu bağladığında dolar — hiçbir otomatik/örtük eşleme yoktur.

## Heading/yaw farkı

`compass_heading`, `yaw` ve `gimbal_heading` üç ayrı alandır ve hiçbiri
diğerinden türetilmez:

- `compass_heading` — manyetik/gerçek pusula yönü.
- `yaw` — aracın gövde eksenindeki dönüş açısı.
- `gimbal_heading` — kameranın baktığı yön.

İkisi karışıyorsa (örn. kaynak veride tek bir "heading" kolonu varsa),
hangi canonical alana karşılık geldiğine operatör karar vermeli ve yalnız
o alanı manifestte eşlemelidir.

## Altitude datum

`reference: AGL | MSL | WGS84` zorunludur; belirtilmeden manifest
reddedilir. Farklı videolarınız farklı datum kullanıyorsa, tek bir
`altitude_m` eşlemesi tek bir datum'u temsil eder — birden fazla datum'u
tek alanda sessizce birleştirmeyin; gerekiyorsa ayrı bir `extra` alanı ile
ikinci datum'u da (filtrelenemez, yalnız bilgi amaçlı) taşıyabilirsiniz.

## Velocity semantics

`kind: ground_speed | air_speed` zorunludur; belirtilmeden reddedilir.

## Continuous/categorical/circular alan örnekleri

```yaml
altitude_m:            # continuous
  source: alt_agl_m
  reference: AGL
  type: continuous
  interpolation: linear
  aggregation: median

compass_heading:        # circular_deg — yalnız circular/circular_mean kabul eder
  source: heading_deg
  type: circular_deg

flight_mode:             # categorical (extra) — yalnız locf/mode kabul eder
  source: flight_mode
  type: categorical
```

Circular ve categorical alanlar için `interpolation`/`aggregation` uyumsuz
verilirse (örn. categorical alana `interpolation: linear`) manifest
validation'da açıkça reddedilir — sessizce yanlış davranmaz.

## Extra telemetry alanları

`telemetry.extra` altındaki alanlar (örnek: `battery_v`, `link_quality_pct`)
P0 filtre/index/ClickHouse kolonu üretmez, yalnız sonuç detay panelinde
salt-okunur gösterilir ve `run_segment_telemetry.extra` JSONB'sinde saklanır
— kaybolmaz.

## Missing telemetry policy

Bir dataset'te bazı videoların telemetrisi yoksa `pairing.telemetry_glob`'u
opsiyonel bırakın (`manifest_csv` stratejisinde `telemetry_path` boş
bırakılabilir); o video için canonical telemetri alanları `None` kalır,
ingest durmaz. `time_alignment.max_gap_s`, bir pencerenin en yakın telemetri
kaydından ne kadar uzaklaşabileceğini sınırlar — aşılırsa o alan `None`
döner (LOCF/interpolation uygulanmaz).

## Video–telemetri eşleştirmesi deterministik mi?

Evet: `filename_stem` stratejisi dosya adına, `manifest_csv` stratejisi
açık CSV satırına dayanır — ikisi de aynı girdide her zaman aynı eşlemeyi
üretir; rastgelelik veya "en yakın dosyayı tahmin et" mantığı yoktur.

## Üç video ile smoke manifesti

Küçük bir `window.size_s`/`stride_s` ile 2-3 video koyup preflight'ı
çalıştırın:

```bash
python scripts/preflight.py --dataset datasets/kurum.yaml --env-file .env \
  --json-out artifacts/faz11/preflight_smoke.json
```

`status=pass` ve `video_count=3` (veya kaç video koyduysanız) görene kadar
büyük corpus'a geçmeyin.

## Büyük corpus manifesti

Aynı manifest, `videos_glob`'un daha geniş bir desene işaret etmesi
dışında değişmez (`"videos/**/*.mp4"` gibi bir desen zaten tüm alt
dizinleri kapsar). `DECODE_CHUNK_S`, `DECODE_PREFETCH_WINDOWS`,
`EMBED_BATCH_SIZE`, `DB_WRITE_BATCH_SIZE` `.env`'de kalır — manifest
dosya keşfinden bağımsızdır.

## Preflight hata örnekleri

```text
exit 3, "duplicate_video_ids": iki dosya aynı video_id'ye çözülüyor
  -> video_id_from deseninizi daha ayırt edici yapın.

exit 3, "telemetry_alignment": "absolute telemetry clock requires video_start_unix_s"
  -> video_start_time_from ayarınız (filename/container_creation_time/manifest_csv
     kolonu) gerçek veriden anchor çözemiyor; sahte anchor uydurmayın, kaynak
     veriyi düzeltin.

exit 3, "pairing": "telemetry CSV lacks mapped columns: [...]"
  -> telemetry.fields/extra'daki source adları CSV başlığıyla eşleşmiyor.

exit 5, "model_bundle": "bundle model_revision mismatch"
  -> MODEL_BUNDLE_ROOT, .env'deki QWEN_MODEL_REVISION ile eşleşmeyen bir
     bundle'a işaret ediyor.
```
