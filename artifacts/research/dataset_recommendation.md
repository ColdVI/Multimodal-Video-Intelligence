# Dataset karari (notebook 00 - GERCEK calistirmadan uretildi)

Uretim zamani: 2026-07-29T08:07:37.793602+00:00

## Aktif deneye alinan uc dataset (spec SS2.2 ile ayni, degistirilmedi)

1. **AU-AIR** - ana hybrid dataset. Erisilebilirlik denetimi: orijinal
   GitHub Pages/repo artik yok, ama Google Drive dosyalari (spec'teki
   "gdown, 2 Drive linki" yontemiyle) GUNCEL id'lerle dogrulandi ve
   erisilebilir (annotations dosyasi 3.9 MB olarak indi, spec'teki
   boyutla uyusuyor).
2. **CapERA** - ana semantic + MRL dataset. GitHub kaynagi canli.
3. **MSR-VTT 1k-A** - external benchmark. HuggingFace kaynagi canli.

## ALFA (yalnizca masa basi)

MAVLink 2.0 protokolu CANLI kaynaktan dogrulandi (github.com/castacks/alfa-dataset).
Tam CSV kolon-alan eslemesi bu asamada KISMI - gercek sequence dosyasi
gerektiriyor, `future_work.md`'ye yazildi.

## Sonraki adim

Notebook 01: AU-AIR indirme ve dogrulama (SS4.2).
