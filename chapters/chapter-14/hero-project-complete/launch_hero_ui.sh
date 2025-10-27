#!/bin/bash

# Hero Project - Launch Enhanced Gradio UI
# This script launches the final working Hero Project interface

echo "🚀 Starting Hero Project Enhanced Chat Interface..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup_and_test.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if model exists
if [ ! -f "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" ]; then
    echo "❌ Model not found. Please run setup_and_test.sh first to download the model."
    exit 1
fi

# Launch the enhanced UI
echo "🌐 Launching Hero Project Enhanced Chat Interface..."
echo "📍 Access the interface at: http://localhost:7868"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python simple_hero_ui.py
