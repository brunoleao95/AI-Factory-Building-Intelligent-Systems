# =============================================================
# Etapa 3 - Sobe o backend (FastAPI/uvicorn) e o expoe na internet via ngrok.
# Use ANTES de gravar o video / durante a avaliacao.
#
#   1. Garanta que o Ollama esta rodando:   ollama serve
#   2. Configure NGROK_DOMAIN no .env (dominio estatico gratuito do ngrok) e
#      autentique o ngrok uma vez:           ngrok config add-authtoken <TOKEN>
#   3. Rode:                                  ./scripts/start_public.ps1
#
# A URL publica (https://<seu-dominio>.ngrok-free.app) deve ser colocada nos
# Secrets do app no Streamlit Cloud (BACKEND_URL) e no documento da Etapa 3.
# =============================================================
param(
    [int]$Port = 8000,
    [string]$Domain = $env:NGROK_DOMAIN
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "venv nao encontrado em $python. Crie e instale requirements-backend.txt." -ForegroundColor Red
    exit 1
}

# Carrega NGROK_DOMAIN do .env, se nao veio por parametro
if (-not $Domain -and (Test-Path (Join-Path $root ".env"))) {
    $line = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match "^NGROK_DOMAIN=" } | Select-Object -First 1
    if ($line) { $Domain = ($line -split "=", 2)[1].Trim() }
}

Write-Host "Iniciando backend (uvicorn) na porta $Port..." -ForegroundColor Cyan
Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "$Port" `
    -WorkingDirectory $root

Start-Sleep -Seconds 3

if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Host "`nngrok nao encontrado. Instale com:  winget install --id Ngrok.Ngrok" -ForegroundColor Yellow
    Write-Host "Depois autentique:  ngrok config add-authtoken <SEU_TOKEN>" -ForegroundColor Yellow
    Write-Host "Backend segue rodando local em http://127.0.0.1:$Port" -ForegroundColor Green
    exit 0
}

Write-Host "Expondo via ngrok..." -ForegroundColor Cyan
if ($Domain) {
    Write-Host "Dominio estatico: https://$Domain" -ForegroundColor Green
    ngrok http "--domain=$Domain" $Port
} else {
    Write-Host "Sem NGROK_DOMAIN definido - usando URL aleatoria (muda a cada execucao)." -ForegroundColor Yellow
    ngrok http $Port
}
