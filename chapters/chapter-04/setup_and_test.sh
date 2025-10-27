#!/bin/bash

# Chapter 4: Essential Metrics for On-Device AI Performance Setup Script
# On-Device AI: The Small Language Models Revolution

set -e  # Exit on any error

echo "🚀 Setting up Chapter 4: Essential Metrics for On-Device AI Performance"
echo "======================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION+ is required, but you have $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "🔧 Installing Jupyter kernel..."
python -m ipykernel install --user --name=venv --display-name="Python (venv)"

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/benchmarks
mkdir -p data/profiles
mkdir -p data/models

# Test installation
echo "🧪 Testing installation..."

# Test imports
python3 -c "
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
print('✅ All core dependencies imported successfully')
"

# Test system monitoring
python3 -c "
import psutil
cpu_count = psutil.cpu_count()
memory = psutil.virtual_memory()
print(f'✅ System monitoring working: {cpu_count} CPUs, {memory.total // (1024**3)}GB RAM')
"

# Test PyTorch
python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__} working')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Launch Jupyter: jupyter notebook metrics_demo.ipynb"
echo "2. Select 'Python (venv)' kernel in Jupyter"
echo "3. Follow the notebook to explore AI performance metrics"
echo ""
echo "🔧 Troubleshooting:"
echo "- If you see import errors, make sure you're in the venv: source venv/bin/activate"
echo "- If Jupyter doesn't show the venv kernel, run: python -m ipykernel install --user --name=venv --display-name='Python (venv)'"
echo ""
echo "📚 Documentation: README.md"
echo "🎯 Demo: metrics_demo.ipynb"
