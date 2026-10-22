$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "logs\maodou.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "Xiaomaodou is not running."
    exit 0
}

$savedPid = [int](Get-Content $pidFile -ErrorAction Stop)
$process = Get-Process -Id $savedPid -ErrorAction SilentlyContinue

if ($process -and $process.ProcessName -like "python*") {
    Stop-Process -Id $savedPid
    Write-Host "Xiaomaodou stopped. PID: $savedPid"
}

Remove-Item -Path $pidFile -ErrorAction SilentlyContinue

