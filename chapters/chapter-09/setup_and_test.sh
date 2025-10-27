#!/bin/bash

# Chapter 9: Hardware Battlefield - Setup Script
echo "🔧 Setting up Chapter 9: Hardware Battlefield Demo"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "🔧 Installing Jupyter kernel..."
python -m ipykernel install --user --name=venv --display-name="Python 3 (venv)"

echo "✅ Setup complete! Launching Jupyter notebook..."
echo "📓 Opening hardware_battlefield_demo.ipynb"
jupyter notebook --no-browser hardware_battlefield_demo.ipynb
