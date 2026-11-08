$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $projectRoot) {
    $projectRoot = Get-Location
}

# 1. Ensure backend\.env exists
$rootEnv = Join-Path $projectRoot ".env"
$backendEnv = Join-Path $projectRoot "backend\.env"
if ((-not (Test-Path $backendEnv)) -and (Test-Path $rootEnv)) {
    Copy-Item $rootEnv $backendEnv
    Write-Host "[OK] Copied .env to backend/.env"
}

# 2. Check and start FastAPI backend
$backendActive = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $backendActive) {
    Write-Host "[...] Starting FastAPI backend on port 8000..."
    $backendDir = Join-Path $projectRoot "backend"
    Start-Process -FilePath "D:\hindi_rag_venv\Scripts\python.exe" `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $backendDir `
        -WindowStyle Minimized
} else {
    Write-Host "[OK] Backend already running on port 8000."
}

# 3. Check and start Vite frontend
$frontendActive = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue
if (-not $frontendActive) {
    Write-Host "[...] Starting Vite React frontend on port 5173..."
    $frontendDir = Join-Path $projectRoot "frontend"
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "npm run dev" `
        -WorkingDirectory $frontendDir `
        -WindowStyle Minimized
} else {
    Write-Host "[OK] Frontend already running on port 5173."
}

Write-Host "[...] Waiting for servers to become ready..."
$ready = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    $bOk = $false
    $fOk = $false
    try {
        $bResp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($bResp.status -eq "healthy") { $bOk = $true }
    } catch {}

    try {
        $fResp = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 2 -ErrorAction Stop
        if ($fResp.StatusCode -eq 200) { $fOk = $true }
    } catch {}

    if ($bOk -and $fOk) {
        $ready = $true
        Write-Host "[SUCCESS] Both backend and frontend are live!"
        break
    }
}

if (-not $ready) {
    Write-Host "[NOTE] Server startup is taking a few moments. Opening browser now..."
}

Write-Host "[OK] Opening http://localhost:5173 in default browser..."
Start-Process "http://localhost:5173"
