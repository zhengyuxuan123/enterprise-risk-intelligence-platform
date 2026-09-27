# stop-python-agent.ps1
# Stop only the Uvicorn process that serves this project's app.main on port 8081.

[CmdletBinding(SupportsShouldProcess = $true)]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$script:dryRun = $WhatIfPreference
$WhatIfPreference = $false
function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }

$hits = @()
foreach ($p in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)) {
    if ($p.CommandLine -and $p.CommandLine -match '-m\s+uvicorn\s+app\.main:app' -and
        $p.CommandLine -match '--port\s+8081') {
        $hits += [pscustomobject]@{ Pid = [int]$p.ProcessId; Line = [string]$p.CommandLine }
    }
}

Say ''
if (-not $hits) { Say '[i] no Python Agent was found - nothing to stop.' 'Yellow'; Say ''; exit 0 }
Say 'These Python Agent processes will be stopped:' 'Yellow'
foreach ($h in $hits) { Say ('  pid ' + $h.Pid + ' :: ' + $h.Line) }
if ($script:dryRun) { Say '[WhatIf] nothing was stopped.' 'Gray'; exit 0 }
if (-not $Force) {
    $ans = Read-Host 'Type Y to confirm'
    if ($ans -notmatch '^(y|Y)') { Say 'cancelled - nothing was stopped.'; exit 0 }
}
foreach ($h in $hits) {
    try { Stop-Process -Id $h.Pid -Force; Say ('  [OK] stopped pid ' + $h.Pid) 'Green' }
    catch { Say ('  [X] failed to stop pid ' + $h.Pid + ' : ' + $_.Exception.Message) 'Red' }
}
Start-Sleep -Seconds 2
if (Get-NetTCPConnection -State Listen -LocalPort 8081 -ErrorAction SilentlyContinue) {
    Say '[!] port 8081 is still held by something.' 'Yellow'
} else { Say '[OK] port 8081 is free.' 'Green' }
Say ''
