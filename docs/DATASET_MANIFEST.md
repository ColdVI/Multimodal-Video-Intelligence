# Dataset manifest sözleşmesi

Kurum entegrasyonunun tek veri yapılandırma yüzeyi YAML manifestidir. Host
yolları manifestte bulunmaz: `source`, `pairing` ve CSV içindeki göreli yollar
`DATA_ROOT` altında çözülür. Mutlak yol, Windows drive yolu, `..` veya symlink
ile `DATA_ROOT` dışına çıkış reddedilir.

Başlangıç şablonu: `datasets/example_uav.yaml`.

## Dosya bulma ve eşleme

`source.videos_glob` ve `pairing.telemetry_glob`, container içindeki
`/workspace/data` köküne göredir. Basit eşleme için her iki dosyanın stem'i
aynı olmalıdır. Parçalı veya farklı adlandırılmış uçuşlarda
`pairing.strategy: manifest_csv` kullanılır. Eşleme CSV'si şu kolonları taşır:

```text
video_id,video_path,telemetry_path,video_start_unix_s,offset_s
```

CSV yolları da `DATA_ROOT` altında relative olmak zorundadır.

## Zaman hizalama

Absolute telemetri clock'ları (`unix_ms`, `unix_s`, `iso8601`) için:

```text
aligned_s = telemetry_unix_s - video_start_unix_s - offset_s
```

Relative telemetri clock'u (`relative_s`) için:

```text
aligned_s = telemetry_relative_s - offset_s
```

Offset işareti bilinçli olarak aynıdır: pozitif `offset_s`, telemetriyi video
timeline'ında daha erken konuma taşır. Relative clock video wall-clock anchor
aramaz. Absolute clock için container creation time, filename veya pairing CSV
anchor'ı gerçek veriden çözülemezse preflight FAIL olur; sistem sahte anchor
üretmez.

`telemetry_clock: iso8601` içinde timezone-naive bir zaman damgası (örn.
`2026-01-01T12:00:00`, ofset yok) asla sessizce UTC sayılmaz: manifestin
`time_alignment.timezone` alanı (varsayılan `UTC`, ama açık ve kurum
tarafından değiştirilebilir) ile `ZoneInfo` üzerinden yerelleştirilir — bu,
`video_start_time_from: filename` anchor'ının zaten kullandığı yerelleştirme
ile aynı mekanizmadır. Zaman damgasının kendisi açık bir UTC ofseti
taşıyorsa (`+00:00`, `Z` gibi) bu her zaman `timezone` ayarının önüne geçer.
DST dahil timezone kuralları `ZoneInfo`'nun sistem tz veritabanından gelir.

## Canonical telemetri

P0 filtre alanları şunlardır:

```text
event_category, split, video_id, latitude, longitude, altitude_m,
velocity_mps, roll, pitch, yaw, yaw_rate, gimbal_pitch, gimbal_heading,
compass_heading, person_count, vehicle_count, bus_count, is_night
```

- `altitude_m`, `reference: AGL|MSL|WGS84` belirtmeden kabul edilmez.
- `velocity_mps`, `kind: ground_speed|air_speed` belirtmeden kabul edilmez.
- `compass_heading`, `yaw` ve `gimbal_heading` farklı semantiklerdir; manifest
  şemasında hiçbir otomatik eşleme yoktur — bir alan yalnız operatör YAML'da
  o tam canonical ada açıkça bir `source` kolonu bağladığında doldurulur.
- Circular derece alanları `circular` interpolation ve `circular_mean`
  aggregation kullanır; 359° ile 1° ortalaması 180° değildir.
- Categorical alanlar aynı şekilde yalnız `locf` interpolation ve `mode`
  aggregation kabul eder; sayısal bir interpolation/aggregation seçilirse
  manifest validation'da reddedilir.
- Açık birim dönüşümü `scale` ve `offset` ile yapılır; gizli tahmin yoktur.
- Canonical olmayan alanlar `telemetry.extra` altında saklanır ve P0 filtre
  kontrolü/index/ClickHouse kolonu üretmez.

## Proprietary format sınırı

Generic CSV/JSON canonical sözleşmesine dönüştürülmüş kurum verisi Python
çekirdeği değiştirilmeden ingest edilir. `.tlog`, `.bin`, MISB KLV veya kurum
içi binary formatlar için yalnız kaynak formatı canonical record iterator'a
dönüştüren bir adapter gerekir; retrieval, embedding, DB ve UI katmanları
değişmez.

Adapter sözleşmesi:

```python
class TelemetryAdapter(Protocol):
    def iter_records(self, source: Path) -> Iterator[TelemetryRecord]:
        ...
```

## Preflight

Host ve container denetimi hiçbir DB veya dataset yazımı yapmaz:

```bash
python scripts/preflight.py \
  --dataset datasets/kurum.yaml \
  --env-file .env \
  --json-out artifacts/faz11/preflight.json
```

Exit codes:

```text
0 success
2 configuration
3 data/manifest
4 GPU/runtime
5 model bundle
6 disk/resources
```

`not_run`, `pass` değildir. Özellikle video decoder/model bundle henüz hazır
değilse veya gerçek kurum dosyası yoksa artifact nedeni ve gereken komutu açık
taşır.
