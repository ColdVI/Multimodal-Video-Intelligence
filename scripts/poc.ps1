[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        'help', 'download-data', 'infra-up', 'infra-down', 'schema', 'frames', 'windows',
        'detect', 'embed', 'load', 'ingest', 'groundtruth', 'eval',
        'search-report', 'bench', 'fiftyone', 'test'
    )]
    [string] $Task,
    [string] $Model = 'xclip_hf_zeroshot',
    [string[]] $Sequence = @(),
    [switch] $AllModels,
    [switch] $NoLaunch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $repoRoot

function Assert-ExitCode([string] $operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$operation başarısız oldu (exit=$LASTEXITCODE)"
    }
}

function Invoke-Python([string[]] $arguments) {
    if (-not (Test-Path -LiteralPath $python)) {
        throw '.venv bulunamadı. Önce: python -m venv .venv'
    }
    & $python @arguments
    Assert-ExitCode 'Python komutu'
}

function Invoke-Task([string] $name) {
    switch ($name) {
        'help' {
            Write-Output 'Kullanım: .\scripts\poc.ps1 <görev> [-Model model_adı] [-Sequence sekans] [-AllModels] [-NoLaunch]'
            Write-Output 'Görevler: download-data infra-up schema frames windows detect embed load ingest'
            Write-Output '          groundtruth eval search-report bench fiftyone test infra-down'
        }
        'infra-up' {
            & docker compose up -d
            Assert-ExitCode 'docker compose up'
        }
        'download-data' { Invoke-Python @('scripts/download_visdrone.py') }
        'infra-down' {
            & docker compose down
            Assert-ExitCode 'docker compose down'
        }
        'schema' {
            $schema = Get-Content -LiteralPath 'schema.sql' -Raw -Encoding UTF8
            $schema | & docker compose exec -T clickhouse clickhouse-client --multiquery
            Assert-ExitCode 'ClickHouse schema'
        }
        'frames' {
            $arguments = @('ingest/01_frames_to_video.py')
            foreach ($item in $Sequence) {
                $arguments += @('--sequence', $item)
            }
            Invoke-Python $arguments
        }
        'windows' { Invoke-Python @('ingest/02_windowing.py') }
        'detect' { Invoke-Python @('ingest/04_detect.py') }
        'embed' { Invoke-Python @('ingest/03_embed.py', '--model', $Model) }
        'load' { Invoke-Python @('ingest/05_load_clickhouse.py', '--model', $Model) }
        'ingest' {
            Invoke-Task 'frames'
            Invoke-Task 'windows'
            Invoke-Task 'detect'
            Invoke-Task 'embed'
            Invoke-Task 'load'
        }
        'groundtruth' { Invoke-Python @('eval/make_groundtruth.py') }
        'eval' {
            $arguments = @('eval/run_eval.py')
            if (-not $AllModels) {
                $arguments += @('--model', $Model)
            }
            Invoke-Python $arguments
        }
        'search-report' { Invoke-Python @('scripts/run_clickhouse_search_report.py') }
        'bench' { Invoke-Python @('-m', 'bench.runner') }
        'fiftyone' {
            $arguments = @('notebooks/inspect_fiftyone.py')
            if ($NoLaunch) {
                $arguments += '--no-launch'
            }
            Invoke-Python $arguments
        }
        'test' {
            Invoke-Python @('-m', 'pytest', 'tests/', '-v', '-p', 'no:cacheprovider')
        }
    }
}

Invoke-Task $Task
