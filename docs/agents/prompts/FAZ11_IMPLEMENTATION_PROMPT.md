# FAZ11 Implementation Prompt

Bu prompt, repo üzerinde gerçek kanıtla çalışan bir FAZ11 uygulama turu başlatmak içindir.

Başlamadan önce sırasıyla okuyun:

1. [docs/agents/START_HERE.md](../START_HERE.md)
2. [docs/architecture/CURRENT_SYSTEM.md](../../architecture/CURRENT_SYSTEM.md)
3. [docs/operations/STATUS.md](../../operations/STATUS.md)
4. [docs/agents/TASKS.md](../TASKS.md)

Çalışma kuralları:

- Mevcut kullanıcı değişikliklerini koru.
- Doğrulanmamış adımı tamamlandı diye yazma.
- Ağırlık indirme, veri doğrulama, ingest, GT, eval ve raporu ayrı kanıtlarla doğrula.
- Yeni sabitleri config.yaml dışında gömme.
- Son durum için [artifacts/faz11/final_acceptance.json](../../../artifacts/faz11/final_acceptance.json) varsa onu kullan.

Çıktı olarak her turda değişen dosyaları, komutları, kalan riskleri ve tek sonraki adımı raporla.