# run-backend-task.ps1
# Start the packaged Spring Boot jar as a Windows Scheduled Task.
# Why: processes started straight from an automation host (Start-Process / background shell)
# get reaped when the parent command finishes, so "the service is mysteriously gone".
# A scheduled task is hosted by the Task Scheduler service and survives that.
#
# IMPORTANT: this file is pure ASCII - PowerShell 5.1 mis-reads non-ASCII .ps1 without BOM.
#
#   .\run-backend-task.ps1            # (re)create + start + health-check
#   .\run-backend-task.ps1 -Remove    # stop and delete the scheduled task only

[CmdletBinding()]
param(
    [int]$Port = 8080,
    [int]$WaitSeconds = 120,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$root   = $PSScriptRoot
$task   = 'RiskBackend' + $Port
$log    = Join-Path $root '_task-start.txt'
$lines  = New-Object System.Collections.ArrayList
function Say([string]$m) { [void]$lines.Add($m) }

$existing = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if ($existing -and $Remove) {
    Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Unregister-ScheduledTask -TaskName $task -Confirm:$false
    Say ('removed scheduled task ' + $task)
    Set-Content -LiteralPath $log -Value $lines -Encoding UTF8
    exit 0
}

$target = Join-Path $root 'backend\target'
if (-not (Test-Path -LiteralPath $target)) {
    Say ('[X] target folder not found: ' + $target)
    Set-Content -LiteralPath $log -Value $lines -Encoding UTF8
    exit 1
}
$jar = Get-ChildItem -LiteralPath $target -File -Filter 'enterprise-risk-platform-*.jar' |
       Where-Object { $_.Name -notlike '*.original' } |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $jar) {
    Say '[X] no jar found - build first.'
    Set-Content -LiteralPath $log -Value $lines -Encoding UTF8
    exit 1
}

$javaBin = Join-Path $env:JAVA_HOME 'bin\java.exe'
if (-not (Test-Path -LiteralPath $javaBin)) { $javaBin = 'D:\Java17\bin\java.exe' }
if (-not (Test-Path -LiteralPath $javaBin)) { $javaBin = 'java' }

$runLog = Join-Path $target 'app-run.log'
$args   = '-jar "' + $jar.FullName + '" --server.port=' + $Port + ' --logging.file.name="' + $runLog + '"'

if ($existing) {
    Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Unregister-ScheduledTask -TaskName $task -Confirm:$false
    Say ('replaced existing task ' + $task)
}

$action  = New-ScheduledTaskAction -Execute $javaBin -Argument $args -WorkingDirectory $target
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(-5))
$set     = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 365) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Settings $set -Force | Out-Null
Start-ScheduledTask -TaskName $task
Say ('task        : ' + $task)
Say ('jar         : ' + $jar.FullName)
Say ('java        : ' + $javaBin)
Say ('run log     : ' + $runLog)

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$state    = 'timeout'
while ((Get-Date) -lt $deadline) {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($conn) { $state = 'up'; break }
    Start-Sleep -Seconds 2
}
$conn2 = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
Say ('listen pid  : ' + ($(if ($conn2) { $conn2.OwningProcess -join ',' } else { 'none' })))
Say ('state       : ' + $state)
Say ''
Say ('to stop  : schtasks /end /tn "' + $task + '"   (or: .' + '\run-backend-task.ps1 -Remove)')

Set-Content -LiteralPath $log -Value $lines -Encoding UTF8
exit $(if ($state -eq 'up') { 0 } else { 1 })
