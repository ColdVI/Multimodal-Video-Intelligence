param()

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $repoRoot
$outputPath = Join-Path $repoRoot 'artifacts\verify_faz7_output.txt'

Start-Transcript -LiteralPath $outputPath -Force | Out-Null
try {
    "[faz7] verify started: $([DateTimeOffset]::Now.ToString('o'))"
    if (-not (Test-Path -LiteralPath '.env.faz7')) {
        Copy-Item -LiteralPath '.env.faz7.example' -Destination '.env.faz7'
    }

    docker compose -f docker-compose.faz7.yml up -d --build
    if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }

    $health = $null
    foreach ($attempt in 1..60) {
        try {
            $health = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 5
            if ($health.status -eq 'ok' -and $health.embedding_mode -eq 'synthetic') { break }
        } catch {
            $health = $null
        }
        Start-Sleep -Seconds 2
    }
    if ($null -eq $health -or $health.status -ne 'ok') { throw 'API did not become healthy' }
    $health | ConvertTo-Json -Depth 6

    docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
    if ($LASTEXITCODE -ne 0) { throw 'AU-AIR ingestion failed' }

    $body = @{
        query = 'kalabalik trafik'; dataset_id = 'auair'; backend = 'clickhouse'
        strategy = 'prefilter'; dimension = 512; top_k = 10; repeats = 10
    } | ConvertTo-Json -Compress
    $search = Invoke-RestMethod -Uri 'http://localhost:8000/search' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 180
    if ($search.embedding_mode -ne 'synthetic' -or $search.diagnostics.returned_count -ne 10) {
        throw 'search response contract failed'
    }
    $search | ConvertTo-Json -Depth 8

    $stats = Invoke-RestMethod -Uri 'http://localhost:8000/stats' -TimeoutSec 30
    $auair = $stats.datasets | Where-Object dataset_id -eq 'auair'
    foreach ($backend in 'pgvector','clickhouse','qdrant') {
        if ($auair.$backend.'512' -ne 1866) { throw "$backend row parity failed" }
    }
    $stats | ConvertTo-Json -Depth 8

    $ui = Invoke-WebRequest -Uri 'http://localhost:7860' -UseBasicParsing -TimeoutSec 30
    if ($ui.StatusCode -ne 200) { throw 'UI HTTP check failed' }
    '[faz7] UI HTTP 200'
    "[faz7] verify completed: $([DateTimeOffset]::Now.ToString('o'))"
} finally {
    Stop-Transcript | Out-Null
}

