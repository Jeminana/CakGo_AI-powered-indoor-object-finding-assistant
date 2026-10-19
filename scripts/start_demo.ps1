<#
.SYNOPSIS
  Start the whole CakGo demo with one command: backend + frontend dev server + HTTPS tunnel.

.DESCRIPTION
  1. Checks .venv, GPU, cloudflared and free ports.
  2. Starts the backend with the chosen model and refuses to continue if it came up in mock mode.
  3. Warms the model up with one test request.
  4. Starts frontend/dev_server.py and a Cloudflare quick tunnel, and prints the phone link.
  5. Stops everything (including the tunnel) when you press Enter or close the window.

  Logs go to %TEMP%\cakgo_demo\.

.PARAMETER Model
  gdino (default: Grounding DINO, multi-prompt, the team's chosen model), yolo (fine-tuned v1) or mock.

.PARAMETER NoTunnel
  Local only (http://localhost:5173 on this computer); no phone link.

.PARAMETER Check
  Start everything, send one test request through the phone link, then stop. Use it to verify a setup.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Check
  powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Model yolo -NoTunnel
#>
param(
    [ValidateSet("gdino", "yolo", "mock")] [string]$Model = "gdino",
    [switch]$NoTunnel,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$LogDir = Join-Path $env:TEMP "cakgo_demo"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Started = @()

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Fail($text) { Say "[fail] $text" Red; throw $text }

function Stop-All {
    foreach ($p in $Started) {
        if ($p -and -not $p.HasExited) {
            # /T also stops the child process that the .venv python launcher starts
            & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
        }
    }
    Say "[stop] backend, dev server and tunnel stopped. The phone link no longer works." Yellow
}

function Wait-Http($url, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { return Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 } catch { Start-Sleep -Milliseconds 700 }
    }
    return $null
}

function Test-PortFree($port) {
    $used = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return -not $used
}

try {
    # ---------------------------------------------------------------- checks
    $Python = Join-Path $Repo ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) { Fail ".venv not found. Create it first (see detection/experiments/yolo/README.md, Setup)." }
    foreach ($port in 8000, 5173) {
        if (-not (Test-PortFree $port)) { Fail "port $port is already in use. Close the old backend/dev server first." }
    }
    $cuda = & $Python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cuda -ne "True") {
        if ($Model -eq "gdino") { Say "[warn] no CUDA GPU: Grounding DINO will take about 15 s per frame on CPU." Yellow }
        else { Say "[warn] no CUDA GPU: running on CPU." Yellow }
    } else { Say "[ok] GPU available" Green }

    $Cloudflared = $null
    if (-not $NoTunnel) {
        $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
        if ($cmd) { $Cloudflared = $cmd.Source }
        elseif (Test-Path "C:\Program Files (x86)\cloudflared\cloudflared.exe") { $Cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe" }
        else { Fail "cloudflared not found. Install it: winget install --id Cloudflare.cloudflared (or use -NoTunnel)." }
    }

    # ---------------------------------------------------------------- backend
    switch ($Model) {
        "gdino" {
            $env:CAKGO_DETECTOR = "custom"
            $env:CAKGO_DETECTOR_FACTORY = "detection.experiments.grounding_dino.detector:create_detector"
        }
        "yolo" {
            $env:CAKGO_DETECTOR = "yolo"
            $env:CAKGO_YOLO_WEIGHTS = "detection/experiments/yolo/training_runs/yolo26s_cakgo4_v1/weights/best.pt"
            if (-not (Test-Path $env:CAKGO_YOLO_WEIGHTS)) { Fail "best.pt missing at $env:CAKGO_YOLO_WEIGHTS (get it from the team Drive)." }
        }
        "mock" { $env:CAKGO_DETECTOR = "mock" }
    }
    Say "[start] backend ($Model) ..." Cyan
    $Started += Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000" `
        -RedirectStandardOutput "$LogDir\backend.out.log" -RedirectStandardError "$LogDir\backend.err.log" -NoNewWindow -PassThru
    $health = Wait-Http "http://127.0.0.1:8000/api/health" 120
    if (-not $health) { Fail "backend did not start. See $LogDir\backend.err.log" }
    $mode = $health.Headers["X-CakGo-Detector"]
    $expected = @{ gdino = "custom"; yolo = "yolo"; mock = "mock" }[$Model]
    if ($mode -ne $expected) { Fail "backend runs in '$mode' mode, expected '$expected'." }
    Say "[ok] backend running, detector mode: $mode" Green

    # ---------------------------------------------------------------- dev server
    Say "[start] frontend dev server ..." Cyan
    $Started += Start-Process -FilePath $Python -ArgumentList "frontend/dev_server.py" `
        -RedirectStandardOutput "$LogDir\devserver.out.log" -RedirectStandardError "$LogDir\devserver.err.log" -NoNewWindow -PassThru
    if (-not (Wait-Http "http://127.0.0.1:5173/" 30)) { Fail "dev server did not start. See $LogDir\devserver.err.log" }
    Say "[ok] dev server: http://localhost:5173" Green

    Say "[warm-up] first request loads the model (can take 10-60 s) ..." Cyan
    & $Python scripts/demo_request.py --url http://127.0.0.1:5173
    if ($LASTEXITCODE -ne 0) { Fail "test request failed. See $LogDir\backend.err.log" }

    # ---------------------------------------------------------------- tunnel
    $PhoneUrl = $null
    if (-not $NoTunnel) {
        Say "[start] HTTPS tunnel ..." Cyan
        $tunnelLog = "$LogDir\tunnel.err.log"
        if (Test-Path $tunnelLog) { Remove-Item $tunnelLog }
        $Started += Start-Process -FilePath $Cloudflared -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://localhost:5173" `
            -RedirectStandardOutput "$LogDir\tunnel.out.log" -RedirectStandardError $tunnelLog -NoNewWindow -PassThru
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline -and -not $PhoneUrl) {
            Start-Sleep -Seconds 1
            if (Test-Path $tunnelLog) {
                $log = Get-Content $tunnelLog -Raw
                $found = [regex]::Match($log, "https://[a-z0-9-]+\.trycloudflare\.com")
                if ($found.Success -and $log.Contains("Registered tunnel connection")) { $PhoneUrl = $found.Value }
            }
        }
        if (-not $PhoneUrl) { Fail "tunnel did not start. See $tunnelLog" }
        # A new tunnel address needs a few seconds before DNS can find it, so retry for up to a minute.
        Say "[test] request through the phone link (a new link can take up to a minute) ..." Cyan
        $reachable = $false
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline -and -not $reachable) {
            & $Python scripts/demo_request.py --url $PhoneUrl --timeout 20 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { $reachable = $true } else { Start-Sleep -Seconds 3 }
        }
        if ($reachable) {
            & $Python scripts/demo_request.py --url $PhoneUrl --timeout 20
        } elseif ($Check) {
            Fail "the phone link is not reachable after 60 s. See $tunnelLog"
        } else {
            Say "[warn] the link is not reachable yet; wait a little and reload on the phone." Yellow
        }
    }

    Say ""
    Say "=============================================================" Green
    if ($PhoneUrl) {
        Say " Phone link (open in Safari / Chrome, not in KakaoTalk):" Green
        Say "   $PhoneUrl" White
    }
    Say " On this computer:  http://localhost:5173" Green
    Say " Model: $Model    Logs: $LogDir" Green
    Say "=============================================================" Green
    if ($PhoneUrl) { Say " Anyone with the link can use this laptop's detector. Don't post it publicly." Yellow }

    if ($Check) {
        Say "[check] everything works. Stopping (-Check mode)." Green
    } else {
        Read-Host "`nPress Enter to stop the demo"
    }
}
finally {
    Stop-All
    Remove-Item Env:CAKGO_DETECTOR, Env:CAKGO_DETECTOR_FACTORY, Env:CAKGO_YOLO_WEIGHTS -ErrorAction SilentlyContinue
}
