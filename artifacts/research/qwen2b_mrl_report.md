# Notebook 02 - Qwen3-VL-Embedding-2B GPU asamasi sonucu

## GPU durumu
gpu_available=False (hardware_profile=local-cpu)

## embedding_ready bayraklari (dataset basina, birbirinden BAGIMSIZ)
{
  "auair": false,
  "capera": false,
  "msrvtt": false,
  "visdrone": false
}

## MRL turetme ozeti
{
  "auair": {
    "status": "ATLANDI - 2048d checkpoint yok"
  },
  "capera": {
    "status": "ATLANDI - 2048d checkpoint yok"
  },
  "msrvtt": {
    "status": "ATLANDI - 2048d checkpoint yok"
  },
  "visdrone": {
    "status": "ATLANDI - 2048d checkpoint yok"
  }
}

## Checkpoint yollari (resume destekli - hucreyi tekrar calistirmak kaldigi yerden devam eder)
- AU-AIR: artifacts\research\checkpoints\auair_qwen2048.ndjson
- CapERA: artifacts\research\checkpoints\capera_qwen2048.ndjson
- MSR-VTT: artifacts\research\checkpoints\msrvtt_qwen2048.ndjson
- VisDrone: artifacts\research\checkpoints\visdrone_qwen2048.ndjson
