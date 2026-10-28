# Starts FastAPI (8001), Node backend (5000) and the Next.js frontend (3000) in separate PowerShell windows.
# EDIT THIS: path to your trained models (the folder that contains fundus_classification and fundus_segmentation).
param(
    [string]$ModelStore = "C:\Users\mhjha\Documents\GitHub\J26-IT-323\Backend\model_store"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path $ModelStore)) {
    Write-Warning "Model store not found: $ModelStore  (the inference service will report all models unavailable)"
}

$venvActivate = Join-Path $root "inference_service\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Python venv not found. Follow README section 2 first."
    exit 1
}

$inference = "Set-Location '$root\inference_service'; & '$venvActivate'; `$env:MODEL_STORE_DIR = '$ModelStore'; py -m uvicorn app.main:app --host 127.0.0.1 --port 8001"
$backend   = "Set-Location '$root\backend'; npm start"
$frontend  = "Set-Location '$root\frontend'; npm run dev"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $inference
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backend
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontend

Write-Host "Started. Open http://localhost:3000"
Write-Host "Model status: http://127.0.0.1:8001/models"
