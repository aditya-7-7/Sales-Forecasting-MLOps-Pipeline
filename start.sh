#!/bin/bash
# Sales Forecasting MLOps - Startup Script
# This script starts the project and handles network configuration

set -e

echo "🚀 Starting Sales Forecasting MLOps Platform..."

# Step 1: Start Astro development environment
echo "📦 Starting Airflow services with Astronomer..."
astro dev start

# Step 2: Wait for services to be ready
echo "⏳ Waiting for Airflow services to initialize..."
sleep 10

# Step 3: Detect the Airflow network name
echo "🔍 Detecting Airflow network..."
AIRFLOW_NETWORK=$(docker network ls --filter "name=airflow" --format "{{.Name}}" | head -1)

if [ -z "$AIRFLOW_NETWORK" ]; then
    echo "❌ Could not find Airflow network. Make sure 'astro dev start' completed successfully."
    exit 1
fi

echo "✅ Found Airflow network: $AIRFLOW_NETWORK"

# Step 4: Export the network name and start additional services
export AIRFLOW_NETWORK=$AIRFLOW_NETWORK

echo "📦 Starting MinIO, MLflow, and Streamlit services..."
docker compose -f docker-compose.services.yml up -d

# Step 5: Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Step 6: Display access information
echo ""
echo "✅ All services started successfully!"
echo ""
echo "📊 Access your services at:"
echo "   • Airflow UI:     http://localhost:8080 (admin/admin)"
echo "   • MLflow UI:      http://localhost:5001"
echo "   • MinIO Console:  http://localhost:9002 (minioadmin/minioadmin)"
echo "   • Streamlit UI:   http://localhost:8501"
echo ""
echo "🎯 Next steps:"
echo "   1. Open Airflow UI at http://localhost:8080"
echo "   2. Login with admin/admin"
echo "   3. Find 'sales_forecasting_training' DAG"
echo "   4. Click the play button to start training"
echo ""
