$ErrorActionPreference = "SilentlyContinue"

$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "logs\maodou.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "Xiaomaodou is stopped."
    exit 0
}

$savedPid = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
$process = Get-Process -Id $savedPid -ErrorAction SilentlyContinue

if ($process -and $process.ProcessName -like "python*") {
    Write-Host "Xiaomaodou is running. PID: $savedPid"
} else {
    Write-Host "Xiaomaodou is stopped."
}

