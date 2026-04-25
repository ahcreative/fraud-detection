@echo off
REM ============================================================
REM run_pipeline.bat
REM One-click pipeline runner for Windows 11
REM Double-click this file or run from CMD
REM ============================================================

SETLOCAL

SET DATA_DIR=.\data
SET OUTPUT_DIR=.\outputs
SET STRATEGY=smote
SET SAMPLE=0.3

echo.
echo ==================================================
echo   IEEE CIS Fraud Detection Pipeline
echo   Windows 11 ^| Core i5 ^| 16GB RAM
echo ==================================================
echo.

REM Check venv
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Run setup.bat first!
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Check data files
IF NOT EXIST "%DATA_DIR%\train_transaction.csv" (
    echo.
    echo [ERROR] Missing: %DATA_DIR%\train_transaction.csv
    echo Place your Kaggle dataset files in .\data\
    pause
    exit /b 1
)
IF NOT EXIST "%DATA_DIR%\train_identity.csv" (
    echo [ERROR] Missing: %DATA_DIR%\train_identity.csv
    pause
    exit /b 1
)
echo [OK] Data files found

echo.
echo Starting pipeline...
echo   Data      : %DATA_DIR%
echo   Output    : %OUTPUT_DIR%
echo   Strategy  : %STRATEGY%
echo   Sample    : %SAMPLE% (30%% of data for speed)
echo.
echo This will take ~15-25 minutes on Core i5 / 16GB RAM
echo.

python scripts\run_local.py ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --imbalance_strategy %STRATEGY% ^
    --sample_frac %SAMPLE% ^
    --compare_strategies true ^
    --skip_shap false

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   Pipeline Complete!
echo ==================================================
echo.
echo Outputs saved to: %OUTPUT_DIR%
echo.
echo To start monitoring:
echo   docker-compose up -d
echo   Then: http://localhost:3000  (Grafana)
echo.
pause
