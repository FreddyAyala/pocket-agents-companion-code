#!/bin/bash
# Setup and test script for Chapter 7 fine-tuning example

echo "🔧 Setting up Chapter 7: Fine-Tuning & Adaptation Demo"
echo "========================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ from python.org"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Test imports
echo "🧪 Testing imports..."
python3 -c "
import torch
import transformers
import datasets
import peft
import accelerate
import jupyter
print('✅ All imports successful!')
print(f'   PyTorch: {torch.__version__}')
print(f'   Transformers: {transformers.__version__}')
print(f'   Datasets: {datasets.__version__}')
print(f'   PEFT: {peft.__version__}')
print(f'   Accelerate: {accelerate.__version__}')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi

# Install Jupyter kernel for the virtual environment
echo "🔧 Installing Jupyter kernel..."
python3 -m ipykernel install --user --name=venv --display-name="Python (venv)"
if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Could not install Jupyter kernel"
    echo "   You may need to manually select the kernel in Jupyter"
fi

echo ""
echo "✅ Chapter 7 setup complete!"
echo "🚀 Launching Jupyter notebook..."
echo "📓 The notebook will open in your browser automatically"
echo "💡 To run again: source venv/bin/activate && jupyter notebook"
echo ""

# Start Jupyter notebook
jupyter notebook fine_tuning_demo.ipynb