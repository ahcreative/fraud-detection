@echo off
REM ============================================================
REM start_monitoring.bat
REM Starts Prometheus + Grafana + API monitoring stack
REM ============================================================

echo.
echo ==================================================
echo   Starting Fraud Detection Monitoring Stack
echo ==================================================
echo.

REM Check Docker
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo [OK] Docker Desktop running

REM Check if model exists in serving/
IF NOT EXIST "serving\model.pkl" (
    echo.
    echo [WARNING] No model found in .\serving\
    echo The API will start but predictions will return 503.
    echo Run the pipeline first: run_pipeline.bat
    echo.
    echo To copy model manually after pipeline:
    echo   copy outputs\07_serving\model.pkl serving\
    echo   copy outputs\07_serving\preprocessor.pkl serving\
    echo   copy outputs\07_serving\feature_config.json serving\
    echo.
)

REM Start the stack
echo Starting services...
docker-compose up -d

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] docker-compose failed.
    echo Check that docker-compose.yml exists and Docker Desktop is running.
    pause
    exit /b 1
)

echo.
echo Waiting 15 seconds for services to start...
timeout /t 15 /nobreak >nul

echo.
echo Checking service status...
docker-compose ps

echo.
echo ==================================================
echo   Monitoring Stack Running!
echo ==================================================
echo.
echo   API          : http://localhost:8000/health
echo   Prometheus   : http://localhost:9090
echo   Grafana      : http://localhost:3000
echo                  Login: admin / admin123
echo   Alertmanager : http://localhost:9093
echo.
echo   To send a test prediction:
echo   curl -X POST http://localhost:8000/predict ^
echo     -H "Content-Type: application/json" ^
echo     -d "{\"TransactionAmt\": 100, \"ProductCD\": \"W\"}"
echo.
echo   To stop: docker-compose down
echo.
pause
