# Sales Forecasting MLOps - Windows Startup Script
# Run this in PowerShell from the project directory

Write-Host ""
Write-Host "Starting Sales Forecasting MLOps Platform..." -ForegroundColor Cyan

# Step 1: Check Docker is running
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Docker is running" -ForegroundColor Green

# Step 2: Start Astro development environment (Airflow only)
Write-Host ""
Write-Host "Starting Airflow services with Astronomer..." -ForegroundColor Yellow
astro dev start

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start Astro. Check the error above." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Airflow started successfully" -ForegroundColor Green

# Step 3: Wait for Airflow to initialize
Write-Host ""
Write-Host "Waiting for Airflow services to initialize - 20 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# Step 4: Detect the Airflow network name
Write-Host ""
Write-Host "Detecting Airflow network..." -ForegroundColor Yellow
$networks = docker network ls --filter "name=airflow" --format "{{.Name}}"
$networkList = $networks -split "`n"
$AIRFLOW_NETWORK = ""
foreach ($net in $networkList) {
    $trimmed = $net.Trim()
    if ($trimmed -ne "" -and $trimmed -like "*airflow*") {
        $AIRFLOW_NETWORK = $trimmed
        break
    }
}

if ([string]::IsNullOrEmpty($AIRFLOW_NETWORK)) {
    Write-Host "ERROR: Could not find Airflow network." -ForegroundColor Red
    Write-Host "Available networks:" -ForegroundColor Yellow
    docker network ls
    exit 1
}

Write-Host "[OK] Found Airflow network: $AIRFLOW_NETWORK" -ForegroundColor Green

# Step 5: Set environment variable and start additional services
$env:AIRFLOW_NETWORK = $AIRFLOW_NETWORK

Write-Host ""
Write-Host "Starting MinIO, MLflow, and Streamlit services..." -ForegroundColor Yellow
docker compose -f docker-compose.services.yml up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start additional services." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Additional services started" -ForegroundColor Green

# Step 6: Wait for services to be healthy
Write-Host ""
Write-Host "Waiting for all services to be ready - 60 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 60

# Step 7: Check container status
Write-Host ""
Write-Host "Container Status:" -ForegroundColor Yellow
docker ps --format "table {{.Names}}\t{{.Status}}"

# Step 8: Display access information
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  All services started successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access your services at:" -ForegroundColor Cyan
Write-Host "  Airflow UI:     http://localhost:8080  (admin/admin)"
Write-Host "  MLflow UI:      http://localhost:5001"
Write-Host "  MinIO Console:  http://localhost:9002  (minioadmin/minioadmin)"
Write-Host "  Streamlit UI:   http://localhost:8501"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open Airflow UI at http://localhost:8080"
Write-Host "  2. Login with admin/admin"
Write-Host "  3. Find sales_forecasting_training DAG"
Write-Host "  4. Click the play button to start training"
Write-Host ""
