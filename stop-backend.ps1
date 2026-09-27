# stop-backend.ps1
# Stop the packaged Spring Boot jar started by start-backend.ps1.
#
# The match is done on the process command line, so IDEA, DataGrip and the
# Maven embedder (all java.exe too) are left untouched.
#
#   .\stop-backend.ps1           # shows the list, asks for confirmation
#   .\stop-backend.ps1 -Force    # no prompt
#   .\stop-backend.ps1 -WhatIf   # only show what would be stopped

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# Remember -WhatIf, then silence the preference locally: otherwise loading the
# CIM module prints dozens of "setting alias" WhatIf lines before we say anything.
$script:dryRun = $WhatIfPreference
$WhatIfPreference = $false

function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }

# "-jar <something>enterprise-risk-platform-<version>.jar"
$pattern = '-jar\s+"?[^"]*enterprise-risk-platform-[\d.]+\.jar'

$hits = @()
foreach ($p in (Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue)) {
    if ($p.CommandLine -and ($p.CommandLine -match $pattern)) {
        $hits += [pscustomobject]@{ Pid = [int]$p.ProcessId; Line = [string]$p.CommandLine }
    }
}

Say ''
if (-not $hits) {
    Say '[i] no running backend jar was found - nothing to stop.' 'Yellow'
    Say ''
    exit 0
}

Say 'These processes will be stopped:' 'Yellow'
Say ''
foreach ($h in $hits) { Say ('  pid ' + $h.Pid + ' :: ' + $h.Line) }

if ($script:dryRun) {
    Say ''
    Say '[WhatIf] nothing was stopped.' 'Gray'
    Say ''
    exit 0
}

if (-not $Force) {
    Say ''
    $ans = Read-Host 'Type Y to confirm'
    if ($ans -notmatch '^(y|Y)') {
        Say 'cancelled - nothing was stopped.' 'Gray'
        Say ''
        exit 0
    }
}

Say ''
foreach ($h in $hits) {
    try {
        Stop-Process -Id $h.Pid -Force -ErrorAction Stop
        Say ('  [OK] stopped pid ' + $h.Pid) 'Green'
    } catch {
        Say ('  [X] failed to stop pid ' + $h.Pid + ' : ' + $_.Exception.Message) 'Red'
    }
}

Start-Sleep -Seconds 3
Say ''
if (Get-NetTCPConnection -State Listen -LocalPort 8080 -ErrorAction SilentlyContinue) {
    Say '[!] port 8080 is still held by something.' 'Yellow'
} else {
    Say '[OK] port 8080 is free.' 'Green'
}
Say ('     tip: now you can run  mvn package  or start it from IDEA again.') 'Gray'
Say ''
