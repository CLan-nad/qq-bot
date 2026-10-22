# 小毛豆机器人部署脚本的公共函数：参数校验、远端脚本渲染、SSH 执行。
# 被 deploy.ps1 / rollback.ps1 引用，不要单独运行。

$script:DeployDefaults = @{
    Server       = "ubuntu@124.222.100.81"
    IdentityFile = Join-Path $env:USERPROFILE ".ssh\xiaomaodou_deploy_ed25519"
    RemoteDir    = "/opt/maodou"
    ServiceName  = "maodou"
    RepoUrl      = "https://github.com/CLan-nad/qq-bot.git"
    Branch       = "main"
}

function Test-SafeArg {
    param([string]$Name, [string]$Value)
    # 这些值会被拼进远端 bash 脚本，先做字符白名单，杜绝注入
    if ($Value -notmatch '^[\w./:@+~-]+$') {
        throw "参数 ${Name} 含不安全字符，只允许字母数字和 . / : @ + ~ -"
    }
    return $Value
}

function Get-RemoteScriptBody {
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [Parameter(Mandatory)][hashtable]$Tokens
    )
    if (-not (Test-Path -LiteralPath $ScriptPath)) { throw "找不到远端脚本：$ScriptPath" }
    # 必须强制按 UTF-8 读：Windows PowerShell 默认按 ANSI 解析无 BOM 的 UTF-8 文件
    $body = [System.IO.File]::ReadAllText($ScriptPath, [System.Text.Encoding]::UTF8)
    foreach ($key in $Tokens.Keys) {
        $body = $body.Replace("__" + $key + "__", [string]$Tokens[$key])
    }
    if ($body -match '__[A-Z_]+__') {
        throw "远端脚本里还有未替换的占位符：$($Matches[0])"
    }
    return $body
}
function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgList
    )
    # 手工拼命令行并用 Process API 分别捕获两个流：
    # 1) 远端命令含空格时必须整体加引号，PowerShell 的参数传递做不到可控
    # 2) Windows PowerShell 会把外部命令的 stderr 当异常抛出，这里绕开它
    $dq = [string][char]34
    $bs = [string][char]92
    $parts = foreach ($a in $ArgList) {
        if ($a -match ("[" + $dq + "s]")) { $dq + ($a -replace $dq, ($bs + $dq)) + $dq }
        else { $a }
    }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = ($parts -join " ")
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $false
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $proc = [System.Diagnostics.Process]::Start($psi)
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    return [pscustomobject]@{
        ExitCode = $proc.ExitCode
        Stdout   = $outTask.Result
        Stderr   = $errTask.Result
    }
}

function Get-SshArgs {
    param([string]$IdentityFile)
    if (-not $IdentityFile) { $IdentityFile = $script:DeployDefaults.IdentityFile }
    if (-not (Test-Path -LiteralPath $IdentityFile)) { throw "找不到 SSH 私钥：$IdentityFile" }
    return @(
        "-i", $IdentityFile,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15"
    )
}
function Invoke-RemoteScript {
    param(
        [Parameter(Mandatory)][string]$Body,
        [string]$Server,
        [string]$IdentityFile,
        [string]$Label = "run",
        [switch]$AsRoot
    )
    if (-not $Server) { $Server = $script:DeployDefaults.Server }
    # 渲染后的脚本先落到 /tmp 再执行：日志可追溯，失败时能在服务器上原样复跑
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $localPath = Join-Path $env:TEMP ("maodou-" + $Label + "-" + $PID + "-" + $stamp + ".sh")
    [System.IO.File]::WriteAllText($localPath, ($Body -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
    $remotePath = "/tmp/maodou-" + $Label + "-" + $stamp + ".sh"

    $sshArgs = Get-SshArgs -IdentityFile $IdentityFile
    $copy = Invoke-Native -FilePath "scp" -ArgList ($sshArgs + @($localPath, ($Server + ":" + $remotePath)))
    if ($copy.ExitCode -ne 0) {
        Write-Host ("    上传脚本失败：" + $copy.Stderr.Trim()) -ForegroundColor Red
        return $copy.ExitCode
    }

    $runner = "bash " + $remotePath
    if ($AsRoot) { $runner = "sudo -n " + $runner }
    $run = Invoke-Native -FilePath "ssh" -ArgList ($sshArgs + @($Server, ($runner + " 2>&1")))
    foreach ($line in ($run.Stdout -split "`r?`n")) {
        if ($line.Trim().Length -gt 0) { Write-Host "    $line" }
    }
    if ($run.Stderr.Trim().Length -gt 0) {
        Write-Host ("    [ssh] " + $run.Stderr.Trim()) -ForegroundColor DarkGray
    }
    Write-Host ("    （远端脚本留存于 " + $remotePath + "）") -ForegroundColor DarkGray
    return $run.ExitCode
}
function Get-RemoteStateValue {
    param([string]$Key, [string]$Server, [string]$IdentityFile)
    if (-not $Server) { $Server = $script:DeployDefaults.Server }
    if ($Key -notmatch "^[a-z-]+$") { throw "非法的 state 键名：$Key" }
    $sshArgs = Get-SshArgs -IdentityFile $IdentityFile
    $run = Invoke-Native -FilePath "ssh" -ArgList ($sshArgs + @($Server, "cat /var/lib/maodou-deploy/$Key 2>/dev/null || true"))
    return $run.Stdout.Trim()
}

function Show-RemoteExitMeaning {
    param([int]$Code)
    $map = @{
        3 = "服务器工作树和仓库不一致（可能有历史手改）。人工确认后加 -Force 重跑。"
        4 = "pip 安装失败，已自动回滚到部署前的版本。"
        5 = "单元测试未通过，服务没有重启，已自动回滚。"
        6 = "服务重启后健康检查没通过，已自动回滚。"
        9 = "致命错误（进不了目录 / git 操作失败 / 目标版本不存在）。"
    }
    if ($map.ContainsKey($Code)) { Write-Host $map[$Code] -ForegroundColor Red }
}