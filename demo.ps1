# Composite Inspection - one-command demo.
#
#     .\demo.ps1          start everything, open the browser, Ctrl+C stops it all
#     .\demo.ps1 -Stop    just kill anything left on the demo ports
#
# Starts the inference backend (port 8000) and the app (port 5173), waits until both
# actually answer, then opens the browser. The report assistant additionally needs
# Ollama; the script detects it and says so either way rather than failing.

param([switch]$Stop)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$py = "C:\Users\vkant\.conda\envs\uwe_hack\python.exe"

function Stop-Ports {
    $c = Get-NetTCPConnection -LocalPort 5173, 8000 -State Listen -ErrorAction SilentlyContinue
    if ($c) {
        $c | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            Write-Host "stopping pid $_"
            Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
    }
}

Stop-Ports                       # a stale server from a previous run must never win the port
if ($Stop) { Write-Host "demo ports clear."; exit 0 }

if (-not (Test-Path $py)) { Write-Error "uwe_hack python not found at $py" }
if (-not (Test-Path "$repo\frontend\public\data\fixtures.json")) {
    Write-Host "fixtures.json missing - building the preloaded samples first (a few minutes on GPU)..."
    & $py "$repo\frontend\tools\build_fixtures.py"
}

Write-Host "starting inference backend (8000) and app (5173)..."
$api = Start-Process -FilePath $py -ArgumentList "`"$repo\frontend\server\app.py`"" `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru
$web = Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" `
        -WorkingDirectory "$repo\frontend" -WindowStyle Hidden -PassThru

# Wait until both genuinely answer - the backend loads 155MB of weights, so a fixed
# sleep would either waste time or open the browser onto a dead page.
function Wait-Http($url, $name, $seconds) {
    for ($i = 0; $i -lt $seconds * 2; $i++) {
        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
            Write-Host "  $name up"
            return $true
        } catch { Start-Sleep -Milliseconds 500 }
    }
    Write-Warning "$name did not answer at $url"
    return $false
}
$apiUp = Wait-Http "http://127.0.0.1:8000/api/health" "backend" 60
$webUp = Wait-Http "http://127.0.0.1:5173/" "app" 60
if (-not ($apiUp -and $webUp)) { Stop-Ports; Write-Error "startup failed - see above" }

try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null
    Write-Host "  ollama up - report assistant is live"
} catch {
    Write-Host "  ollama not running - everything works except the report assistant" `
        "(optional: 'ollama serve' in another terminal)"
}

Start-Process "http://127.0.0.1:5173/inspections"
Write-Host ""
Write-Host "Demo running. Ctrl+C here stops both servers." -ForegroundColor Green
try {
    Wait-Process -Id $web.Id
} finally {
    # Ctrl+C lands here: take the servers down with the script, no orphans.
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $web.Id -Force -ErrorAction SilentlyContinue
    Stop-Ports
    Write-Host "stopped."
}
