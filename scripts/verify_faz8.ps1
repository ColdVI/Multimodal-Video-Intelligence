param()

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $repoRoot
$outputPath = Join-Path $repoRoot 'artifacts/verify_faz8_output.txt'

Start-Transcript -LiteralPath $outputPath -Force | Out-Null
try {
    "[faz8] verify started: $([DateTimeOffset]::Now.ToString('o'))"
    if (-not (Test-Path -LiteralPath '.env.faz7')) {
        Copy-Item -LiteralPath '.env.faz7.example' -Destination '.env.faz7'
    }
    docker compose -f docker-compose.faz7.yml up -d --build
    if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }

    docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
    if ($LASTEXITCODE -ne 0) { throw 'AU-AIR ingestion failed' }

    & .venv\Scripts\python.exe scripts\readiness_check.py --profile system --strict
    if ($LASTEXITCODE -ne 0) { throw 'system readiness failed' }

    $env:RUN_FAZ8_INTEGRATION = '1'
    & .venv\Scripts\python.exe -m pytest -q service\tests -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw 'Faz 8 service tests failed' }

    $env:PYTHONPATH = 'service'
    & .venv\Scripts\python.exe -m app.bench.matrix --suite all --quick --out artifacts\research\test_matrix_all.csv
    if ($LASTEXITCODE -ne 0) { throw 'Faz 8 matrix failed' }

    & .venv\Scripts\python.exe scripts\readiness_check.py --profile quality --json
    "[faz8] quality readiness is informational until Colab artifacts and cached ingest exist"
    "[faz8] verify completed: $([DateTimeOffset]::Now.ToString('o'))"
} finally {
    Stop-Transcript | Out-Null
}
