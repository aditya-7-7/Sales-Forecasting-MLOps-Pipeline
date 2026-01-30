# Sales Forecasting MLOps - Windows Stop Script
# Run this in PowerShell to stop all services

Write-Host ""
Write-Host "Stopping Sales Forecasting MLOps Platform..." -ForegroundColor Yellow

# Step 1: Stop additional services (MinIO, MLflow, Streamlit)
Write-Host "Stopping MinIO, MLflow, and Streamlit services..." -ForegroundColor Yellow
docker compose -f docker-compose.services.yml down 2>$null

# Step 2: Stop Astro services
Write-Host "Stopping Airflow services..." -ForegroundColor Yellow
astro dev stop

Write-Host ""
Write-Host "[OK] All services stopped!" -ForegroundColor Green
Write-Host ""
