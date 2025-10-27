#!/bin/bash

# Chapter 14: Agentic Best Practices Setup Script
# On-Device AI: The Small Language Models Revolution

set -e  # Exit on any error

echo "🚀 Setting up Chapter 14: Agentic Best Practices"
echo "=============================================="

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
mkdir -p data/agents
mkdir -p data/tools
mkdir -p data/workflows

# Test installation
echo "🧪 Testing installation..."

# Test imports
python3 -c "
import requests
import json
import numpy as np
import pandas as pd
print('✅ All core dependencies imported successfully')
"

# Test agentic frameworks
python3 -c "
try:
    import openai
    print('✅ OpenAI client available')
except ImportError:
    print('⚠️ OpenAI client not available (optional)')

try:
    import langchain
    print('✅ LangChain available')
except ImportError:
    print('⚠️ LangChain not available (optional)')
"

# Test web scraping capabilities
python3 -c "
import requests
from bs4 import BeautifulSoup
print('✅ Web scraping tools available')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Launch Jupyter: jupyter notebook agentic_patterns_demo.ipynb"
echo "2. Select 'Python (venv)' kernel in Jupyter"
echo "3. Follow the notebook to explore agentic AI patterns"
echo ""
echo "🔧 Troubleshooting:"
echo "- If you see import errors, make sure you're in the venv: source venv/bin/activate"
echo "- If Jupyter doesn't show the venv kernel, run: python -m ipykernel install --user --name=venv --display-name='Python (venv)'"
echo ""
echo "📚 Documentation: README.md"
echo "🎯 Demo: agentic_patterns_demo.ipynb"
