#!/bin/bash

# Quick Setup Script for Loan Prediction App

echo ""
echo "======================================"
echo "Loan Prediction App - Quick Setup"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

echo "[1/4] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi
echo "[✓] Dependencies installed"

echo ""
echo "[2/4] Training the model..."
python3 pipeline.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to train model"
    exit 1
fi
echo "[✓] Model trained and saved"

echo ""
echo "[3/4] Verifying model files..."
if [ ! -f "models/rf_model.pkl" ]; then
    echo "Error: rf_model.pkl not found"
    exit 1
fi
if [ ! -f "models/label_encoders.pkl" ]; then
    echo "Error: label_encoders.pkl not found"
    exit 1
fi
if [ ! -f "models/feature_names.pkl" ]; then
    echo "Error: feature_names.pkl not found"
    exit 1
fi
echo "[✓] All model files verified"

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "======================================"
echo "To run the Streamlit app, execute:"
echo "  streamlit run app.py"
echo "======================================"
echo ""
