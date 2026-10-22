# 把服务器回滚到上一个版本（或指定版本）。
#   .\scripts\rollback.ps1                 回到 deploy.ps1 记录的上一个版本
#   .\scripts\rollback.ps1 -Rev <sha>      回到指定版本
# 复用的是 deploy-remote.sh：切代码 -> 依赖对齐 -> 备份 -> 重启 -> 健康检查。
[CmdletBinding()]
param(
    [string]$Rev = "",
    [string]$Server = "",
    [string]$IdentityFile = "",
    [string]$RemoteDir = "",
    [string]$ServiceName = "",
    [string]$RepoUrl = ""
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "deploy-common.ps1")

if (-not $Server) { $Server = $script:DeployDefaults.Server }
if (-not $IdentityFile) { $IdentityFile = $script:DeployDefaults.IdentityFile }
if (-not $RemoteDir) { $RemoteDir = $script:DeployDefaults.RemoteDir }
if (-not $ServiceName) { $ServiceName = $script:DeployDefaults.ServiceName }
if (-not $RepoUrl) { $RepoUrl = $script:DeployDefaults.RepoUrl }
$Branch = $script:DeployDefaults.Branch

if ($Rev) {
    $target = $Rev
}
else {
    $target = Get-RemoteStateValue -Key "prev-rev" -Server $Server -IdentityFile $IdentityFile
    if (-not $target) { throw "/var/lib/maodou-deploy/prev-rev 是空的，说明还没用 deploy.ps1 部署过；请显式给 -Rev" }
}
foreach ($pair in @(@("Server", $Server), @("RemoteDir", $RemoteDir), @("ServiceName", $ServiceName),
                   @("RepoUrl", $RepoUrl), @("Rev", $target))) {
    [void](Test-SafeArg -Name $pair[0] -Value $pair[1])
}
Write-Host ("准备回滚到 {0}" -f $target) -ForegroundColor Yellow
$body = Get-RemoteScriptBody -ScriptPath (Join-Path $PSScriptRoot "deploy-remote.sh") -Tokens @{
    REMOTE_DIR = $RemoteDir; SERVICE = $ServiceName; REPO_URL = $RepoUrl; BRANCH = $Branch
    TARGET     = $target;    FORCE   = "0";          SKIP_TESTS = "1"
}
$code = Invoke-RemoteScript -Body $body -Server $Server -IdentityFile $IdentityFile -Label "rollback" -AsRoot
if ($code -ne 0) {
    Show-RemoteExitMeaning -Code $code
    Write-Host ("回滚未完成，退出码 {0}" -f $code) -ForegroundColor Red
    exit $code
}
Write-Host ("回滚完成，当前线上 {0}" -f (Get-RemoteStateValue -Key "rev" -Server $Server -IdentityFile $IdentityFile)) -ForegroundColor Green