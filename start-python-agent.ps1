# start-python-agent.ps1
# Start the Python agent service (FastAPI on 127.0.0.1:8081).
# Paths are resolved from this script's own folder, so it works from anywhere.
#
# IMPORTANT: keep this file pure ASCII. PowerShell 5.1 reads .ps1 as ANSI/GBK when
# there is no BOM, so any non-ASCII character here corrupts parsing.
#
#   .\start-python-agent.ps1                  # start in background, wait until healthy
#   .\start-python-agent.ps1 -Foreground      # run attached, Ctrl+C to stop
#   .\start-python-agent.ps1 -Port 8082       # use another port
#   .\start-python-agent.ps1 -WaitSeconds 30  # shorter health-check wait

[CmdletBinding()]
param(
    [int]$Port = 8081,
    [switch]$Foreground,
    [int]$WaitSeconds = 60
)

$ErrorActionPreference = 'Stop'
$root    = $PSScriptRoot
$appDir  = Join-Path $root 'pyagent'

function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }

Say ''
Say '=== Python agent launcher ===' 'Cyan'
Say ''

if (-not (Test-Path -LiteralPath (Join-Path $appDir 'app\main.py'))) {
    Say ('[X] pyagent\app\main.py not found under: ' + $appDir) 'Red'
    exit 1
}

# --- pick an interpreter -------------------------------------------------
# Order matters: a project-local .venv first (that is what README asks for),
# then the managed workbuddy venv, then whatever "python" resolves to.
$py = $null
foreach ($cand in @(
        (Join-Path $appDir '.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE '.workbuddy\binaries\python\envs\default\Scripts\python.exe'),
        'python')) {
    if ($cand -eq 'python') {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $py = 'python'; break }
    } elseif (Test-Path -LiteralPath $cand) { $py = $cand; break }
}
if (-not $py) {
    Say '[X] no python interpreter found. Create one:' 'Red'
    Say '    cd pyagent; python -m venv .venv; .venv\Scripts\pip install -r requirements.txt' 'Yellow'
    exit 1
}
Say ('python   : ' + $py)

# --- load .env if present ------------------------------------------------
# Keeps one source of truth for config: pyagent\.env is the same key set as
# backend\.env, so both sides can share a file without "the difference was
# just a different config" surprises.
$envFile = Join-Path $appDir '.env'
if (Test-Path -LiteralPath $envFile) {
    $loaded = 0
    $loadedAiKey = $false
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        $t = $line.Trim()
        if ($t.Length -eq 0 -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -le 0) { continue }
        $k = $t.Substring(0, $i).Trim()
        $v = $t.Substring($i + 1).Trim()
        # Strip surrounding quotes. Empty model values intentionally clear an
        # inherited stale model so runtime discovery can choose a valid one.
        if ($v.Length -ge 2 -and $v[0] -eq '"' -and $v[-1] -eq '"') { $v = $v.Substring(1, $v.Length - 2) }
        if ($v.Length -ge 2 -and $v[0] -eq "'" -and $v[-1] -eq "'") { $v = $v.Substring(1, $v.Length - 2) }
        if ($v.Length -eq 0) {
            if ($k -in @('AI_MODEL', 'AI_MODEL_CANDIDATES', 'AI_API_KEY', 'OPENAI_API_KEY')) {
                Remove-Item -Path ("Env:" + $k) -ErrorAction SilentlyContinue
                $loaded++
            }
            continue
        }
        Set-Item -Path ("Env:" + $k) -Value $v
        if ($k -eq 'AI_API_KEY') { $loadedAiKey = $true }
        $loaded++
    }
    Say ('env file : ' + $envFile + '  (' + $loaded + ' keys)')
} else {
    Say 'env file : (none - using process environment; copy pyagent\.env.example to .env)' 'Yellow'
}

if ($env:PY_AGENT_PORT -and $env:PY_AGENT_PORT -ne "$Port") {
    Say ('[!] .env sets PY_AGENT_PORT=' + $env:PY_AGENT_PORT + ' but -Port is ' + $Port + '; -Port wins on the command line.') 'Yellow'
}

# --- AI summary (same rationale as start-backend.ps1) --------------------
Say ''
$aiBase = $env:AI_BASE_URL
if (-not $aiBase) { $aiBase = '(unset -> ' + 'https://ark.cn-beijing.volces.com/api/v3)' }
$aiModel = $env:AI_MODEL
if (-not $aiModel) { $aiModel = '(unset -> auto-pick from account models)' }
Say ('AI base-url : ' + $aiBase)
Say ('AI model    : ' + $aiModel)
if ($env:AI_API_KEY -and $env:AI_API_KEY.Trim().Length -gt 0) {
    Say 'AI key      : AI_API_KEY is set (value hidden)'
    if ($loadedAiKey) {
        Say 'AI key source: pyagent\.env' 'Green'
    } else {
        Say 'AI key source: inherited process environment (not persisted in pyagent\.env)' 'Yellow'
        Say '               A stale shell can therefore keep using an old or revoked key.' 'Yellow'
    }
} else {
    Say 'AI key      : AI_API_KEY is EMPTY - model calls will degrade to the local report' 'Yellow'
}

# --- port check ----------------------------------------------------------
$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) {
    $owner = ($busy | Select-Object -First 1).OwningProcess
    Say ''
    Say ('[!] port ' + $Port + ' is already in use by pid ' + $owner + ' - nothing was started.') 'Yellow'
    exit 1
}

$logDir = Join-Path $root 'qa'
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir 'pyagent-run.log'

if ($Foreground) {
    Say ''
    Say ('starting in FOREGROUND on 127.0.0.1:' + $Port + ' - press Ctrl+C to stop') 'Green'
    Say ''
    Push-Location -LiteralPath $appDir
    & $py -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    Pop-Location
    exit $LASTEXITCODE
}

Say ''
Say ('starting in background on 127.0.0.1:' + $Port + ' ...')

# NOTE: do not use Start-Process -RedirectStandardOutput here -- on this machine
# PowerShell 5.1 throws "duplicate dictionary key" when the environment has
# case-duplicate vars. Redirect inside the child shell instead.
try {
    $proc = Start-Process -FilePath $py `
                          -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port") `
                          -WorkingDirectory $appDir `
                          -WindowStyle Hidden -PassThru
} catch {
    Say ''
    Say ('[X] failed to launch: ' + $_.Exception.Message) 'Red'
    exit 1
}

Say ('pid      : ' + $proc.Id)
Say ('log      : ' + $log + '  (redirect there yourself, or run -Foreground)')
Say 'waiting for the service to answer ...'

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$state    = 'timeout'
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { $state = 'exited'; break }
    try {
        $r = Invoke-WebRequest -Uri ("http://127.0.0.1:$Port/healthz") -UseBasicParsing -TimeoutSec 4
        if ($r.StatusCode -eq 200) { $state = 'up'; break }
    } catch { }
    Start-Sleep -Seconds 1
}

Say ''
switch ($state) {
    'up' {
        Say ('[OK] python agent is UP  ->  http://127.0.0.1:' + $Port) 'Green'
        Say '     self-checks          ->  /api/ai/health  /api/ai/agents  /api/ai/migration' 'Green'
        Say ('     to stop              ->  Stop-Process -Id ' + $proc.Id) 'Green'
    }
    'exited' {
        Say '[X] the process exited early - run with -Foreground to see the traceback.' 'Red'
    }
    default {
        Say ('[!] no answer after ' + $WaitSeconds + 's - run with -Foreground to see why.') 'Yellow'
    }
}
Say ''
