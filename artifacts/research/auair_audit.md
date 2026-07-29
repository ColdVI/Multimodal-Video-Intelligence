# AU-AIR indirme ve dogrulama - denetim raporu (GERCEK calistirmadan)

Uretim zamani: 2026-07-29T08:23:43.537120+00:00

## Lisans duzeltmesi
Spec SS2.1: "CC BY 4.0" (YANLIS). Gercek veri: **CC BY-NC-SA** (Attribution-
NonCommercial-ShareAlike). Bu notebook'un ticari olmayan arastirma
kullanimini engellemiyor, ama karar raporuna DOGRU lisansla tasindi.

## Sema
Gercek alan adlari (VARSAYILMADI, canli veriden okundu): `image_name`,
`time{year,month,day,hour,min,sec,ms}` - `hour/min/sec` bir video icinde
SABIT (baslangic zaman damgasinin kopyasi), `ms` ise 1000'in cok ustune
cikan (ör. 741800), dakika basindan itibaren GECEN TOPLAM MILISANIYE -
gercek zaman `datetime(y,m,d,h,dakika) + timedelta(milliseconds=ms)` ile
kuruldu (ilk denemede `ms` mikrosaniye sanilmisti, bu HATALIYDI - yukarida
duzeltildi). `longtitude` (spec'te "lo" - gercekte bu yazim hatali ama
orijinal alan adi), `latitude`, `altitude` (**milimetre**, spec'in
varsaydigi metre DEGIL - /1000 ile donusturuldu), `linear_x/y/z` (spec'in
varsaydigi Vx/Vy/Vz), `angle_phi/theta/psi` (roll/pitch/yaw),
`bbox[{top,left,height,width,class}]`.

Toplam annotation: 32823 (spec'in tahmini 32.283'e
yakin, birebir degil).

## Video rekonstruksiyonu
- Δt-histogram yontemi (spec SS4.2 adim4, GAP_FACTOR=10, DUZELTILMIS zaman formuluyle): 90 video.
- Dosya-adi-oneki capraz kontrolu (14 hane, x/xx alt-parcalari BIRLESTIRILDI - ms-sureklilik
  kaniti bunlarin ayni kesintisiz ucusun parcalari oldugunu gosterdi): 8 video.
- Kullanilan ana yontem: dosya-adi-oneki (8 video) - deterministik,
  saat yuvarlama gurultusune tabi degil (SS11 karari, manifestte kayitli).

## HARD STOP kapisi
{
  "n_videos_in_range_6_12": true,
  "n_videos": 8,
  "min_eff_fps_over_2": true,
  "min_eff_fps_observed": 2.5272087067861717,
  "altitude_valid_ratio_over_0.95": true,
  "altitude_valid_ratio": 1.0
}

Sonuc: GECTI

## Uretilen artifactlar
- artifacts\research\auair_segments.parquet (1866 segment/pencere)
- artifacts\research\auair_telemetry.parquet (1866 telemetri satiri)
- artifacts\research\selectivity_thresholds.json
- artifacts\research\auair_errors.jsonl (81 hata/uyari - atilmadi)

## Bilinen sinirlama
Goruntu piksel verisi (images.zip, ~2.2 GB) bu oturumda TAMAMEN indirilemedi
(gozlenen baglanti hizi ~0.1-0.3 MB/s). Bu, YUKARIDAKI DOGRULAMALARIN
HICBIRINI ETKILEMEDI (hepsi annotations.json metadata'sindan calisti).
Piksel verisi yalnizca notebook 02'nin Qwen embedding uretiminde gerekli
olurdu - o adim zaten bu makinede GPU kapisinda ayri bir nedenle duruyor
(bkz. notebook 02). "images.zip KISMI (indirme arka planda devam ediyor/yarim kaldi)" olarak isaretlendi,
"tamamlandi" DENMEDI.
