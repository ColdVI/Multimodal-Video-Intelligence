# Notebooks

## Primary entrypoint

[`production/VideoSearch_Unified_Runner.ipynb`](production/VideoSearch_Unified_Runner.ipynb)

Bu notebook kurum datasetinin:

- yerel klasör, ZIP, opsiyonel Google Drive veya URL'den alınması,
- video/CSV profiling,
- source-to-canonical telemetry mapping,
- production manifest üretimi,
- production data preflight,
- onboarding bundle export,
- opsiyonel Qwen portable embedding,
- aynı hosttaki Docker FAZ11 sistemine `ingest --resume`

akışlarını tek dosyada birleştirir.

Google Drive zorunlu değildir.

## Specialist notebook

[`08_colab_portable_runner.ipynb`](08_colab_portable_runner.ipynb), yalnız Drive tabanlı Qwen embedding üretimi için hazırlanmış uzman akıştır. Unified Runner normal kullanıcılar için ana giriş noktasıdır.

Diğer numaralı notebook'lar araştırma, benchmark veya tarihsel POC amaçlıdır; production kurulum kaynağı olarak kullanılmamalıdır.
