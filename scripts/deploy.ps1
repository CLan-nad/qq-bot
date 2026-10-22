# 一键部署小毛豆机器人到生产服务器。
#   .\scripts\deploy.ps1              部署 origin/main 最新提交
#   .\scripts\deploy.ps1 -DryRun      只探测服务器状态，不动线上
#   .\scripts\deploy.ps1 -Force       确认放弃服务器上的手改后再部署
[CmdletBinding()]
param(
    [string]$Ref = "origin/main",
    [string]$Server = "",
    [string]$IdentityFile = "",
    [string]$RemoteDir = "",
    [string]$ServiceName = "",
    [string]$RepoUrl = "",
    [switch]$SkipTests,
    [switch]$Force,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "deploy-common.ps1")

if (-not $Server) { $Server = $script:DeployDefaults.Server }
if (-not $IdentityFile) { $IdentityFile = $script:DeployDefaults.IdentityFile }
if (-not $RemoteDir) { $RemoteDir = $script:DeployDefaults.RemoteDir }
if (-not $ServiceName) { $ServiceName = $script:DeployDefaults.ServiceName }
if (-not $RepoUrl) { $RepoUrl = $script:DeployDefaults.RepoUrl }
$Branch = $script:DeployDefaults.Branch

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
if (-not (Test-Path -LiteralPath $IdentityFile)) { throw "找不到 SSH 私钥：$IdentityFile" }
foreach ($pair in @(@("Server", $Server), @("RemoteDir", $RemoteDir),
                   @("ServiceName", $ServiceName), @("RepoUrl", $RepoUrl), @("Ref", $Ref))) {
    [void](Test-SafeArg -Name $pair[0] -Value $pair[1])
}

function Get-Git { param([string[]]$GitArgs)
    $r = Invoke-Native -FilePath "git" -ArgList (@("-C", $repoRoot) + $GitArgs)
    if ($r.ExitCode -ne 0) { throw ("git " + ($GitArgs -join " ") + " 失败：" + $r.Stderr.Trim()) }
    return $r.Stdout.Trim()
}
Write-Host "== 1/4 本地检查 ==" -ForegroundColor Cyan
$porcelain = Get-Git @("status", "--porcelain")
if ($porcelain) {
    throw ("本地工作树有未提交改动，先 commit 再部署：`n" + $porcelain)
}
$sha = Get-Git @("rev-parse", "--verify", ("${Ref}^{commit}"))
$subject = Get-Git @("log", "-1", "--format=%s", $sha)
Write-Host ("   目标提交 {0}  {1}" -f $sha.Substring(0, 7), $subject)
$ancestor = Invoke-Native -FilePath "git" -ArgList @("-C", $repoRoot, "merge-base", "--is-ancestor", $sha, "origin/$Branch")
if ($ancestor.ExitCode -ne 0) { throw "目标提交还不在 origin/$Branch 上，先 git push 再部署" }
Write-Host "   已推送到 origin/$Branch"

# 服务器到 github.com 的 git 传输通道时通时断（api.github.com 通、github.com 超时），
# 原写法一次 fetch 不动就整个部署作废。这里顺手把 bundle 传到服务器当后备：
# 正常部署根本不会用它，只有远端 fetch origin 失败时才拿它补齐对象。
$bundleLocal = Join-Path $env:TEMP "maodou-deploy-bundle.bundle"
$bundleRemote = "/tmp/maodou-deploy-bundle.bundle"
if (Test-Path -LiteralPath $bundleLocal) { [System.IO.File]::Delete($bundleLocal) }
[void](Get-Git @("bundle", "create", $bundleLocal, $Ref))
if (-not (Test-Path -LiteralPath $bundleLocal)) { throw "bundle 生成失败：$bundleLocal" }
$bundleKB = [math]::Round((Get-Item -LiteralPath $bundleLocal).Length / 1KB)
$copyBundle = Invoke-Native -FilePath "scp" -ArgList (
    (Get-SshArgs -IdentityFile $IdentityFile) + @($bundleLocal, ($Server + ":" + $bundleRemote))
)
if ($copyBundle.ExitCode -ne 0) {
    Write-Host ("   bundle 上传失败（不影响走 origin 部署）：" + $copyBundle.Stderr.Trim()) -ForegroundColor DarkGray
} else {
    Write-Host ("   后备 bundle 已上传：{0} KB" -f $bundleKB)
}

$tokens = @{
    REMOTE_DIR = $RemoteDir
    SERVICE    = $ServiceName
    REPO_URL   = $RepoUrl
    BRANCH     = $Branch
    TARGET     = $sha
    FORCE      = $(if ($Force) { "1" } else { "0" })
    SKIP_TESTS = $(if ($SkipTests) { "1" } else { "0" })
}

Write-Host "== 3/4 服务器探测 ==" -ForegroundColor Cyan
$probeBody = Get-RemoteScriptBody -ScriptPath (Join-Path $PSScriptRoot "probe-remote.sh") -Tokens @{
    REMOTE_DIR = $RemoteDir; SERVICE = $ServiceName; BRANCH = $Branch; TARGET = $sha
}
$probeCode = Invoke-RemoteScript -Body $probeBody -Server $Server -IdentityFile $IdentityFile -Label "probe" -AsRoot
if ($probeCode -ne 0) { Write-Host "探测失败，不继续部署" -ForegroundColor Red; exit $probeCode }
if ($DryRun) { Write-Host "-DryRun 结束，线上没有任何改动" -ForegroundColor Green; exit 0 }
Write-Host "== 4/4 执行部署 ==" -ForegroundColor Cyan
$body = Get-RemoteScriptBody -ScriptPath (Join-Path $PSScriptRoot "deploy-remote.sh") -Tokens $tokens
$code = Invoke-RemoteScript -Body $body -Server $Server -IdentityFile $IdentityFile -Label "deploy" -AsRoot
if ($code -ne 0) {
    Show-RemoteExitMeaning -Code $code
    Write-Host ("部署未完成，退出码 {0}" -f $code) -ForegroundColor Red
    exit $code
}
$online = Get-RemoteStateValue -Key "rev" -Server $Server -IdentityFile $IdentityFile
$before = Get-RemoteStateValue -Key "prev-rev" -Server $Server -IdentityFile $IdentityFile
Write-Host ("部署完成：线上 {0}，上一个版本 {1}（需要回滚就跑 rollback.ps1）" -f $online, $before) -ForegroundColor Green