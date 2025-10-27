#!/usr/bin/env python3
"""
Flask API with Real Local LLM
This API uses the actual local language model instead of mock responses.
"""

from flask import Flask, request, jsonify
import time
from local_llm_service import get_llm_service

app = Flask(__name__)

# Get the LLM service
llm_service = get_llm_service()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = llm_service.get_status()
    return jsonify({
        "status": "healthy",
        "llm_loaded": status["loaded"],
        "model_name": status["model_name"],
        "device": status["device"],
        "parameters": status["parameters"],
        "timestamp": time.time()
    })

@app.route('/generate', methods=['POST'])
def generate_text():
    """Text generation endpoint using real LLM"""
    try:
        data = request.get_json()
        
        # Input validation
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400
        
        prompt = data['prompt']
        max_length = data.get('max_length', 100)
        
        # Generate response using real LLM
        start_time = time.time()
        response = llm_service.generate_response(prompt, max_length)
        generation_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "response": response,
            "generation_time": generation_time,
            "model": llm_service.model_name,
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    status = llm_service.get_status()
    return jsonify({
        "models": [{
            "name": status["model_name"],
            "loaded": status["loaded"],
            "device": status["device"],
            "parameters": status["parameters"],
            "type": "on-device"
        }]
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed service status"""
    return jsonify(llm_service.get_status())

if __name__ == '__main__':
    print("🚀 Starting Flask API with Real Local LLM...")
    print("📡 API Endpoints:")
    print("   GET  /health    - Health check")
    print("   POST /generate  - Text generation with real LLM")
    print("   GET  /models    - List models")
    print("   GET  /status    - Detailed status")
    print("\n🌐 API will be available at: http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
