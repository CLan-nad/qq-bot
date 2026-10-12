$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "logs"
$pidFile = Join-Path $logDir "xiaoxia.pid"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $pidFile) {
    $savedPid = [int](Get-Content $pidFile -ErrorAction Stop)
    $existing = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
    if ($existing -and $existing.ProcessName -like "python*") {
        Write-Host "Xiaomaodou is already running. PID: $savedPid"
        exit 0
    }
}

$env:PYTHONUTF8 = "1"
$python = Join-Path $root ".venv\Scripts\python.exe"
$stdout = Join-Path $logDir "xiaoxia.out.log"
$stderr = Join-Path $logDir "xiaoxia.err.log"

$process = Start-Process `
    -FilePath $python `
    -ArgumentList "main.py" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

Set-Content -Path $pidFile -Value $process.Id
Write-Host "Xiaomaodou started. PID: $($process.Id)"

