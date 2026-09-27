# start-backend.ps1
# Start the packaged Spring Boot jar (backend/target/enterprise-risk-platform-*.jar).
# Paths are resolved from this script's own folder, so it works from anywhere.
#
# IMPORTANT: keep this file pure ASCII. PowerShell 5.1 reads .ps1 as ANSI/GBK when
# there is no BOM, so any non-ASCII character here corrupts parsing.
#
#   .\start-backend.ps1                 # start in background, wait until healthy
#   .\start-backend.ps1 -Port 8081      # use another port
#   .\start-backend.ps1 -Foreground     # run attached, Ctrl+C to stop
#   .\start-backend.ps1 -WaitSeconds 30 # shorter health-check wait

[CmdletBinding()]
param(
    [int]$Port = 8080,
    [switch]$Foreground,
    [int]$WaitSeconds = 120
)

$ErrorActionPreference = 'Stop'
$root   = $PSScriptRoot
$target = Join-Path $root 'backend\target'

function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }

Say ''
Say '=== Backend jar launcher ===' 'Cyan'
Say ''

if (-not (Test-Path -LiteralPath $target)) {
    Say ('[X] target folder not found: ' + $target) 'Red'
    Say '    Build first:  mvn -f backend/pom.xml clean package -DskipTests' 'Yellow'
    exit 1
}

$jar = Get-ChildItem -LiteralPath $target -File -Filter 'enterprise-risk-platform-*.jar' |
       Where-Object { $_.Name -notlike '*.original' } |
       Sort-Object LastWriteTime -Descending |
       Select-Object -First 1

if (-not $jar) {
    Say '[X] no enterprise-risk-platform-*.jar found in backend\target - build the project first.' 'Red'
    exit 1
}

Say ('jar      : ' + $jar.FullName)
Say ('built at : ' + $jar.LastWriteTime)
Say ('size     : ' + [math]::Round($jar.Length / 1MB, 1) + ' MB')

# --- AI provider summary -------------------------------------------------
# Purpose: "I set the system env var but it still fails" is almost always one of
#   (a) this shell/process predates the change, or
#   (b) only the Key was changed while base-url still points at another vendor.
# Print what this process will actually use, before it starts.
$aiBase = $env:AI_BASE_URL
if (-not $aiBase) { $aiBase = '(unset -> default DashScope compatible-mode)' }
$aiModel = $env:AI_MODEL
if (-not $aiModel) { $aiModel = '(unset -> default qwen-plus)' }
Say ''
Say ('AI base-url : ' + $aiBase)
Say ('AI model    : ' + $aiModel)
Say 'AI runtime   : requests are forwarded to Python Agent; pyagent\.env is authoritative.' 'Gray'
$anyKey = $false
foreach ($n in @('AI_API_KEY', 'DASHSCOPE_API_KEY', 'OPENAI_API_KEY')) {
    $v = [Environment]::GetEnvironmentVariable($n, 'Process')
    if ($v -and $v.Trim().Length -gt 0) {
        $anyKey = $true
        Say ('AI key      : ' + $n + ' is set (value hidden)')
    }
}
if (-not $anyKey) { Say 'AI key      : NONE of AI_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY is set' 'Yellow' }
if ($aiBase -match 'volces' -and -not $env:AI_API_KEY) {
    Say '[!] base-url points at Volcano Ark but AI_API_KEY is empty:' 'Yellow'
    Say '    the platform will fall back to DASHSCOPE_API_KEY / OPENAI_API_KEY,' 'Yellow'
    Say '    which is a different vendor key and will be rejected with 401.' 'Yellow'
}
Say '    (changed a system env var just now? an already-running process keeps the' 'Gray'
Say '     OLD values - restart this script, or use the in-page provider switcher.)' 'Gray'
# -------------------------------------------------------------------------

$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) {
    $owner = ($busy | Select-Object -First 1).OwningProcess
    Say ''
    Say ('[!] port ' + $Port + ' is already in use by pid ' + $owner + ' - nothing was started.') 'Yellow'
    Say ('    Stop it first:  Stop-Process -Id ' + $owner) 'Yellow'
    exit 1
}

$javaBin = if ($env:JAVA_HOME -and (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME 'bin\java.exe'))) {
               Join-Path $env:JAVA_HOME 'bin\java.exe'
           } else { 'java' }
Say ('java     : ' + $javaBin)

$log = Join-Path $target 'app-run.log'

if ($Foreground) {
    Say ''
    Say ('starting in FOREGROUND on port ' + $Port + ' - press Ctrl+C to stop') 'Green'
    Say ''
    & $javaBin -jar $jar.FullName "--server.port=$Port"
    exit $LASTEXITCODE
}

Say ''
Say ('starting in background on port ' + $Port + ' ...')

# Do NOT use Start-Process -RedirectStandardOutput on this machine.
# When the environment contains case-duplicate keys (e.g. HTTP_PROXY and http_proxy),
# PowerShell 5.1 throws "duplicate dictionary key" and the launch fails outright.
# Workaround: let Spring Boot write its own log file via --logging.file.name.
try {
    # Start-Process flattens ArgumentList into one command line. Quote the JAR
    # explicitly because this project is commonly stored under a path with spaces.
    $jarArg = '"' + $jar.FullName + '"'
    $proc = Start-Process -FilePath $javaBin `
                          -ArgumentList @('-jar', $jarArg, "--server.port=$Port", "--logging.file.name=$log") `
                          -WindowStyle Hidden -PassThru
} catch {
    Say ''
    Say ('[X] failed to launch java: ' + $_.Exception.Message) 'Red'
    Say '    hint: if it mentions a duplicate dictionary key, an environment variable' 'Yellow'
    Say '          has a case-only duplicate. Use -Foreground, or clean that variable.' 'Yellow'
    exit 1
}

Say ('pid      : ' + $proc.Id)
Say ('log      : ' + $log)
Say 'waiting for the service to answer ...'

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$state    = 'timeout'
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { $state = 'exited'; break }
    try {
        $r = Invoke-WebRequest -Uri ("http://localhost:$Port/api/companies") -UseBasicParsing -TimeoutSec 4
        $state = 'up'
        break
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -eq 401 -or $code -eq 403) { $state = 'up'; break }
    }
    Start-Sleep -Seconds 2
}

Say ''
switch ($state) {
    'up' {
        Say ('[OK] backend is UP  ->  http://localhost:' + $Port) 'Green'
        Say ('     web UI           ->  http://localhost:' + $Port) 'Green'
        Say '     login            ->  admin / Admin@123' 'Green'
    }
    'exited' {
        Say '[X] the process exited early - last lines of the log:' 'Red'
        Say ''
        Get-Content -LiteralPath $log -Tail 30 -ErrorAction SilentlyContinue
    }
    default {
        Say ('[!] no answer after ' + $WaitSeconds + 's - last lines of the log:') 'Yellow'
        Say ''
        Get-Content -LiteralPath $log -Tail 30 -ErrorAction SilentlyContinue
    }
}
Say ''
