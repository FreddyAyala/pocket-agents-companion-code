#!/bin/bash
# Setup and test script for Hero Project: On-Device AI Agent

echo "🚀 Setting up Hero Project: On-Device AI Agent with Vision and RAG"
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
import sentence_transformers
import chromadb
import gradio
import jupyter
import yaml
print('✅ All core imports successful!')
print(f'   PyTorch: {torch.__version__}')
print(f'   Transformers: {transformers.__version__}')
print(f'   Sentence Transformers: {sentence_transformers.__version__}')
print(f'   ChromaDB: {chromadb.__version__}')
print(f'   Gradio: {gradio.__version__}')
print(f'   Jupyter: {jupyter.__version__}')
print(f'   PyYAML: {yaml.__version__}')
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
echo "✅ Hero Project setup complete!"
echo "🚀 Launching Jupyter notebook..."
echo "📓 The notebook will open in your browser automatically"
echo "💡 To run again: source venv/bin/activate && jupyter notebook"
echo ""

# Start Jupyter notebook
jupyter notebook notebooks/hero_project_demo.ipynb


