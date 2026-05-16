@echo off
REM Quick Setup Script for Loan Prediction App

echo.
echo ======================================
echo Loan Prediction App - Quick Setup
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo [✓] Dependencies installed

echo.
echo [2/4] Training the model...
python pipeline.py
if errorlevel 1 (
    echo Error: Failed to train model
    pause
    exit /b 1
)
echo [✓] Model trained and saved

echo.
echo [3/4] Verifying model files...
if not exist "models\rf_model.pkl" (
    echo Error: rf_model.pkl not found
    pause
    exit /b 1
)
if not exist "models\label_encoders.pkl" (
    echo Error: label_encoders.pkl not found
    pause
    exit /b 1
)
if not exist "models\feature_names.pkl" (
    echo Error: feature_names.pkl not found
    pause
    exit /b 1
)
echo [✓] All model files verified

echo.
echo [4/4] Setup complete!
echo.
echo ======================================
echo To run the Streamlit app, execute:
echo   streamlit run app.py
echo ======================================
echo.
pause
