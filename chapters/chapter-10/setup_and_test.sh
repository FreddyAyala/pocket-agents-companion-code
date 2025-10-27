#!/bin/bash
# Setup and test script for Chapter 10: The Deployment Playbook

echo "🚀 Setting up Chapter 10: The Deployment Playbook"
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
echo "⬆️️ Upgrading pip..."
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
import gradio as gr
import streamlit
import fastapi
import uvicorn
import psutil
print('✅ All imports successful!')
print(f'   PyTorch: {torch.__version__}')
print(f'   Transformers: {transformers.__version__}')
print(f'   Gradio: {gr.__version__}')
print(f'   Streamlit: {streamlit.__version__}')
print(f'   FastAPI: {fastapi.__version__}')
print(f'   Uvicorn: {uvicorn.__version__}')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi

# Install Jupyter kernel for the virtual environment
echo "🔧 Installing Jupyter kernel..."
python3 -m ipykernel install --user --name=chapter10 --display-name="Python (chapter10)"
if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Could not install Jupyter kernel"
    echo "   You may need to manually select the kernel in Jupyter"
fi

echo ""
echo "✅ Chapter 10 setup complete!"
echo "🚀 Available deployment examples:"
echo "   📱 Universal: python universal_deployment.py"
echo "   🌐 Web: python web_deployment.py"
echo "   📱 Native: python native_deployment.py"
echo "   🏭 Production: python production_deployment.py"
echo "   📓 Interactive: jupyter notebook deployment_playbook_demo.ipynb"
echo ""
echo "💡 Next steps:"
echo "   1. Run: python test_deployment.py"
echo "   2. Explore: jupyter notebook deployment_playbook_demo.ipynb"
echo "   3. Deploy: Follow the examples in each deployment script"
echo ""
