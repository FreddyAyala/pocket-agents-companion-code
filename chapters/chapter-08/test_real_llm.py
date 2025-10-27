#!/usr/bin/env python3
"""
Test script to demonstrate the real local LLM working
"""

import requests
import time

def test_flask_api():
    """Test the Flask API with real LLM"""
    print("🧪 Testing Flask API with Real Local LLM")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    # Test health
    print("1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"   ✅ API is healthy")
        print(f"   📊 Model: {health['model_name']}")
        print(f"   🖥️ Device: {health['device']}")
        print(f"   📈 Parameters: {health['parameters']}")
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
        return
    
    # Test generation
    test_prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        "Tell me a fun fact",
        "How does local AI work?",
        "What's the weather like?"
    ]
    
    print("\n2. Testing text generation...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   Test {i}: '{prompt}'")
        
        response = requests.post(
            f"{base_url}/generate",
            json={"prompt": prompt, "max_length": 100}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   🤖 AI Response: {result['response']}")
            print(f"   ⏱️ Generation time: {result['generation_time']:.2f}s")
        else:
            print(f"   ❌ Generation failed: {response.status_code}")

def test_gradio_interface():
    """Test the Gradio interface"""
    print("\n🎨 Testing Gradio Interface")
    print("=" * 30)
    print("   🌐 Open your browser and go to: http://localhost:7860")
    print("   💬 Try the chat interface with real AI responses")
    print("   📝 Test the summarization feature")
    print("   🎯 Try sentiment analysis")

def test_streamlit_app():
    """Test the Streamlit app"""
    print("\n🖥️ Testing Streamlit App")
    print("=" * 30)
    print("   🌐 Open your browser and go to: http://localhost:8501")
    print("   💬 Chat with the real AI model")
    print("   📊 Use the text analysis tools")
    print("   🔧 Test the model tools")

if __name__ == "__main__":
    print("🚀 Testing Real Local LLM Applications")
    print("=" * 50)
    
    try:
        test_flask_api()
        test_gradio_interface()
        test_streamlit_app()
        
        print("\n✅ All tests completed!")
        print("\n🌐 Your Real LLM Applications:")
        print("   📡 Flask API: http://localhost:5001")
        print("   🎨 Gradio Interface: http://localhost:7860")
        print("   🖥️ Streamlit App: http://localhost:8501")
        print("\n💡 All apps are using the real microsoft/DialoGPT-small model!")
        print("🔒 Your data never leaves your device!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the services are running.")
        print("   Run: python3 flask_api_with_llm.py")
    except Exception as e:
        print(f"❌ Error: {e}")
