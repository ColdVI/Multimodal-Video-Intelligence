# Gorev listesi

Güncel kanıt ve ölçümler: `STATUS.md`.

## Faz 0 — ortam (insan + agent birlikte)
- [x] `.venv` içinde `pip install -r requirements.txt` (Windows/Python 3.14)
- [x] Docker Compose + gerçek ClickHouse schema — iki tablo `SHOW CREATE`
      ile ve insert/query smoke testiyle doğrulandı
- [x] Resmî Task 4 VisDrone-MOT trainsetini indir, SHA-256 ve
      56 sekans/56 annotation/24.201 kare sözleşmesini doğrula

## Faz 1 — ingest (bir model ile uctan uca)
- [x] 5 gerçek smoke sekansında `frames && windows` (691 kare / 7 pencere)
- [x] 5 gerçek smoke sekansında `detect` — 7 özellik satırı
- [x] 5 gerçek smoke sekansında `embed MODEL=xclip_hf_zeroshot` — 7×512d
- [x] 5 gerçek smoke sekansında `load MODEL=xclip_hf_zeroshot` — 7 satır
- [x] Gerçek embedding ile filtreli/filtresiz `otobüsü göster` sorgusu
- [x] Kaynak ölçümünden sonra seçili 5-sekans iki-model ingest
- [ ] Tam 56-sekans ingest (CPU maliyeti nedeniyle ölçümlü/onaylı başlat)
- [x] Elle bir sorgu dene:
      `python -c "from search.query import search; from search.merge import merge_intervals, pretty; print(pretty(merge_intervals(search('otobusu goster', 'xclip_hf_zeroshot'))))"`

## Faz 2 — ground truth + degerlendirme
- [x] 5-sekans manifest için `groundtruth` — 6 sorgu gerçek annotation ile eşleşti
- [ ] 5-10 sekansi FiftyOne'da gozle kontrol et — ozellikle `gt_walking`
      kamera-hareketi yanlis pozitifi var mi bak
- [x] İki-model smoke `eval` — 24 özet + 98 detay satırı (kalite iddiası değil)
- [ ] `make fiftyone` — pred/gt araliklarini yan yana incele

## Faz 3 — ikinci model + filtre A/B
- [x] 5-sekans `embed/load MODEL=siglip2_frameavg` — 7×1152d / 7 satır
- [x] `eval/run_eval.py` her iki modeli de kosturuyor mu dogrula
      (`MODELS` listesi)
- [ ] Sonuc tablosunu kategori kirilimiyla (tekli/hareket/bilesik) incele —
      filtre ACIK bilesik sorguda gercekten kazanc veriyor mu?

## Faz 4 — olcek testi (opsiyonel ama onerilir)
- [ ] Vektorleri sentetik cogaltip 1M/10M satirda filtreli/filtresiz
      sorgu gecikmesini ol (bkz. CONTEXT.md'deki ClickHouse HNSW supesi)

## Her fazin sonunda
- [x] `make test` hala geciyor mu (46/46 regresyon kontrolu)
