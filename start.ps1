# --------------------------------------------------------
# ICARUS-X -- Windows Startup Script (PowerShell)
# Starts: FastAPI backend (8000) + Static Frontend (3000)
# --------------------------------------------------------

$ProjectRoot = Get-Location

Write-Host "------------------------------------------------"
Write-Host "  ICARUS-X -- Space Weather Forecasting System"
Write-Host "------------------------------------------------"

# -- 1. Check .env ---------------------------------------
if (-not (Test-Path ".env")) {
    Write-Host "[!] Creating .env from template..."
    Copy-Item ".env.template" ".env"
}

# -- 2. Ensure directories exist -------------------------
New-Item -ItemType Directory -Force -Path "data", "models", "logs" | Out-Null

# -- 3. Virtual Environment ------------------------------
if (-not (Test-Path "venv")) {
    Write-Host "[!] Creating virtual environment..."
    python -m venv venv
}

# -- 4. Start FastAPI Backend ----------------------------
Write-Host "[OK] Starting Backend on http://localhost:8000" -ForegroundColor Green
$BackendArgs = "-m", "uvicorn", "m5_architect.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"
$BackendProcess = Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList $BackendArgs -PassThru -NoNewWindow

# -- 5. Start Frontend Server ----------------------------
Write-Host "[OK] Starting Frontend on http://localhost:3000" -ForegroundColor Green
$FrontendArgs = "-m", "http.server", "3000", "--bind", "127.0.0.1", "--directory", "frontend"
$FrontendProcess = Start-Process -FilePath "python.exe" -ArgumentList $FrontendArgs -PassThru -NoNewWindow

Write-Host "------------------------------------------------"
Write-Host "  SERVICES ARE LIVE"
Write-Host "  Dashboard: http://localhost:3000"
Write-Host "  API Docs:  http://localhost:8000/docs"
Write-Host "------------------------------------------------"
Write-Host "  Press Ctrl+C to shut down all services"

try {
    while ($true) {
        if ($BackendProcess.HasExited) {
            Write-Host "[!] Backend process has stopped!" -ForegroundColor Red
            break
        }
        if ($FrontendProcess.HasExited) {
            Write-Host "[!] Frontend process has stopped!" -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host "[STOP] Shutting down services..." -ForegroundColor Yellow
    if ($null -ne $BackendProcess -and -not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $FrontendProcess -and -not $FrontendProcess.HasExited) {
        Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "[OK] Shutdown complete."
}
