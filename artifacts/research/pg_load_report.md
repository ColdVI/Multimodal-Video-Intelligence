# PostgreSQL yukleme raporu (GERCEK calistirmadan)

Konteyner: `research_postgres_faz6` (gecici, port 5433 - ana docker-compose.yml
DEGISTIRILMEDI, ana POC'un "Postgres yok" karari BOZULMADI).

## Satir sayilari (kaynak parquet ile birebir dogrulandi)
{
  "datasets": 1,
  "videos": 8,
  "segments": 1866,
  "segment_metadata": 1866,
  "segment_telemetry": 1866
}

## Secicilik dogrulamasi
`derive_thresholds()` (numpy quantile, notebook 01) ile canli Postgres
sorgusunun (`WHERE column < / > esik`) dondurdugu GERCEK satir sayisi
karsilastirildi - 16 satir, tam liste `auair_selectivity_postgres_verification.csv`'de.
Iki kaynak arasindaki fark yalniz kayan-nokta/quantile enterpolasyon
farklarindan kaynaklanabilir (n=8 video x pencere sayisi kucuk oldugu icin
p=0.001 gibi asiri uc seviyelerde 1-2 satirlik sapma beklenir).

## Kapsam disi
Bu notebook `segment_metadata.object_classes/brightness/camera_motion`
kolonlarini DOLDURMADI - bunlar AU-AIR annotation'inin bbox/class alanindan
turetilebilir ama spec'in bu notebook icin istedigi minimal kapsam
(person_count, vehicle_count, telemetri) ile sinirli tutuldu.
