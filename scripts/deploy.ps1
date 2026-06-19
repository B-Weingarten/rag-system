# deploy.ps1 — Part 1: Install Ollama and pull the gemma3:1b model
# Run from the repo root: .\scripts\deploy.ps1

$ErrorActionPreference = "Stop"
$OllamaUrl = "https://ollama.com/download/OllamaSetup.exe"
$InstallerPath = "$env:TEMP\OllamaSetup.exe"
$OllamaHealth = "http://localhost:11434"
$Model = "gemma3:1b"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    [!!] $msg" -ForegroundColor Yellow }

# ── Step 1: Check if Ollama is already installed ──────────────────────────────
Write-Step "Checking for existing Ollama installation..."
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    Write-OK "Ollama already installed at $($ollamaCmd.Source)"
} else {
    # Try winget first (fastest, cleanest on Windows 11)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Step "Installing Ollama via winget..."
        winget install Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "winget install failed, falling back to direct download..."
            $winget = $null
        } else {
            Write-OK "Ollama installed via winget"
        }
    }

    if (-not $winget) {
        # Fall back: use curl.exe (built into Windows 10/11, much faster than Invoke-WebRequest)
        Write-Step "Downloading Ollama installer via curl.exe..."
        & curl.exe -L --progress-bar -o $InstallerPath $OllamaUrl
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Download failed." -ForegroundColor Red; exit 1
        }
        Write-OK "Downloaded to $InstallerPath"

        Write-Step "Installing Ollama (silent)..."
        Start-Process -FilePath $InstallerPath -ArgumentList "/SILENT" -Wait
        Write-OK "Ollama installed"
    }

    # Refresh PATH so 'ollama' is found in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
}

# ── Step 2: Start Ollama server if not already running ────────────────────────
Write-Step "Checking if Ollama server is running..."
$running = $false
try {
    $resp = Invoke-WebRequest -Uri $OllamaHealth -UseBasicParsing -TimeoutSec 3
    if ($resp.StatusCode -eq 200) { $running = $true }
} catch {}

if ($running) {
    Write-OK "Ollama server already running at $OllamaHealth"
} else {
    Write-Step "Starting Ollama server in background..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Write-Warn "Waiting for server to become healthy..."
    $timeout = 30
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        Start-Sleep -Seconds 2
        $elapsed += 2
        try {
            $resp = Invoke-WebRequest -Uri $OllamaHealth -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) { $running = $true; break }
        } catch {}
    }
    if (-not $running) {
        Write-Host "`n[ERROR] Ollama server did not start within $timeout seconds." -ForegroundColor Red
        exit 1
    }
    Write-OK "Ollama server is healthy"
}

# ── Step 3: Pull the model ────────────────────────────────────────────────────
Write-Step "Pulling model: $Model (this may take a few minutes on first run)..."
ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Failed to pull model $Model" -ForegroundColor Red
    exit 1
}
Write-OK "Model $Model is ready"

Write-Host ""
Write-Host "[DONE] Deployment complete. Run: python scripts\verify.py" -ForegroundColor Green
