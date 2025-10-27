#!/bin/bash

echo "🚀 LAUNCHING HERO PROJECT OPENAI API SERVER"
echo "============================================="
echo ""
echo "🌐 This creates an OpenAI-compatible API for our Atomic Agents"
echo "📱 Compatible with:"
echo "   • HuggingFace Chat-UI"
echo "   • oobabooga text-generation-webui"
echo "   • SillyTavern"
echo "   • LM Studio"
echo "   • Any OpenAI-compatible client!"
echo ""
echo "🔧 Installing required packages..."
source venv/bin/activate && pip install fastapi uvicorn pydantic
echo ""
echo "🚀 Starting API server..."
echo "📡 API will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "💡 Use Ctrl+C to stop the server"
echo ""
source venv/bin/activate && python openai_api_server.py
