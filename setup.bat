@echo off
REM ============================================================
REM setup.bat — Setup for Windows 11 (runs in WSL via Docker)
REM ============================================================

echo.
echo ==================================================
echo   Fraud Detection System — Windows Setup
echo ==================================================

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment ...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/5] Installing dependencies ...
pip install --upgrade pip
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)

echo.
echo [3/5] Creating directories ...
if not exist "data"      mkdir data
if not exist "outputs"   mkdir outputs
if not exist "serving"   mkdir serving
if not exist "models"    mkdir models
if not exist "artifacts" mkdir artifacts

echo.
echo [4/5] Compiling Kubeflow pipeline ...
python pipeline\pipeline.py
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Pipeline compile skipped (kfp may need full install)
) ELSE (
    echo    Pipeline compiled: fraud_detection_pipeline.yaml
)

echo.
echo [5/5] Checking Docker Desktop ...
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Docker Desktop not running. Start Docker Desktop first.
) ELSE (
    echo    Docker Desktop: OK
)

echo.
echo ==================================================
echo   Setup Complete!
echo ==================================================
echo.
echo   NEXT STEPS:
echo.
echo   1. Copy dataset files to .\data\
echo      - train_transaction.csv
echo      - train_identity.csv
echo      - test_transaction.csv
echo      - test_identity.csv
echo.
echo   2. Run local pipeline:
echo      python scripts\run_local.py --data_dir .\data --output_dir .\outputs
echo.
echo   3. Start monitoring:
echo      docker-compose up -d
echo      Then open: http://localhost:3000  (Grafana, admin/admin123)
echo               : http://localhost:9090  (Prometheus)
echo.
echo   4. Submit to Kubeflow:
echo      In WSL/terminal: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80
echo      python scripts\submit_kubeflow.py
echo.
pause
