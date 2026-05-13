# Retrieval ablation runner. For each of 4 configs (dense, +rerank, +hybrid,
# +both), launches a fresh uvicorn process with the right env vars, waits for
# /healthz, runs eval.run_eval, then kills the backend. Results land under
# eval/runs/v0.9.0-<config-name>/. Re-run any time the dataset changes.
#
# Usage (from anywhere):
#   powershell -File "d:\AI Projects\PaperPal\backend\eval\run_ablation.ps1"
#
# Each iteration takes ~90 s (startup + 26-question eval). Total ~6 min.

$ErrorActionPreference = "Stop"
$root = "d:\AI Projects\PaperPal\backend"
$py = Join-Path $root ".venv\Scripts\python.exe"

$configs = @(
    @{ name = "v0.9.0-dense";          hybrid = "false"; reranker = "" },
    @{ name = "v0.9.0-rerank-only";    hybrid = "false"; reranker = "cross-encoder/ms-marco-MiniLM-L-12-v2" },
    @{ name = "v0.9.0-hybrid-only";    hybrid = "true";  reranker = "" },
    @{ name = "v0.9.0-hybrid+rerank";  hybrid = "true";  reranker = "cross-encoder/ms-marco-MiniLM-L-12-v2" }
)

function Stop-Backend {
    Get-Process python* -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path -like "*PaperPal*" } |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 800
}

function Wait-Ready {
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $r = Invoke-WebRequest "http://127.0.0.1:8000/healthz" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

foreach ($cfg in $configs) {
    Write-Host "`n=== $($cfg.name) ===" -ForegroundColor Cyan
    Stop-Backend

    $env:PYTHONUTF8 = "1"
    $env:HYBRID_RETRIEVAL = $cfg.hybrid
    $env:RERANKER_MODEL = $cfg.reranker

    $logOut = Join-Path $root "eval\.last-uvicorn.out"
    $logErr = Join-Path $root "eval\.last-uvicorn.err"

    $proc = Start-Process -FilePath $py `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--app-dir", $root `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr

    Write-Host "Started uvicorn pid=$($proc.Id) (hybrid=$($cfg.hybrid), reranker=$($cfg.reranker -ne ''))"
    if (-not (Wait-Ready)) {
        Write-Host "FAILED: backend did not become ready in 60s" -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        continue
    }

    Set-Location $root
    & $py -m eval.run_eval --name $cfg.name

    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

Write-Host "`n=== ablation complete ===" -ForegroundColor Green
