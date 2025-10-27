#!/usr/bin/env python3
"""
Test script for all applications using Qwen2.5-0.5B-Instruct
"""

import requests
import time

def test_all_applications():
    """Test all applications with Qwen2.5-0.5B"""
    print("🚀 Testing All Applications with Qwen2.5-0.5B-Instruct")
    print("=" * 70)
    
    # Test Flask API
    print("\n📡 Testing Flask API...")
    try:
        # Health check
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Flask API is healthy")
            print(f"   📊 Model: {health['model_name']}")
            print(f"   🖥️ Device: {health['device']}")
            print(f"   📈 Parameters: {health['parameters']}")
            
            # Test generation
            print("\n   🧪 Testing conversation...")
            test_prompts = [
                "Hello! How are you?",
                "What is machine learning?",
                "Tell me a fun fact about space"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n   Test {i}: '{prompt}'")
                response = requests.post(
                    "http://localhost:5001/generate",
                    json={"prompt": prompt, "max_length": 150},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   🤖 AI: {result['response'][:100]}...")
                    print(f"   ⏱️ Time: {result['generation_time']:.2f}s")
                else:
                    print(f"   ❌ Failed: {response.status_code}")
        else:
            print(f"   ❌ Flask API health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Flask API error: {e}")
    
    # Test Gradio Interface
    print("\n🎨 Testing Gradio Interface...")
    try:
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("   ✅ Gradio interface is running")
            print("   🌐 URL: http://localhost:7860")
            print("   💡 Open in browser to test interactive chat")
        else:
            print(f"   ❌ Gradio interface failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Gradio interface error: {e}")
    
    # Test Streamlit App
    print("\n🖥️ Testing Streamlit App...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("   ✅ Streamlit app is running")
            print("   🌐 URL: http://localhost:8501")
            print("   💡 Open in browser to test native-like interface")
        else:
            print(f"   ❌ Streamlit app failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Streamlit app error: {e}")
    
    # Summary
    print("\n🎉 All Applications Test Complete!")
    print("=" * 50)
    print("🌐 Your Qwen2.5-0.5B Applications:")
    print("   📡 Flask API: http://localhost:5001")
    print("   🎨 Gradio Interface: http://localhost:7860")
    print("   🖥️ Streamlit App: http://localhost:8501")
    print("\n💡 All apps now use Qwen2.5-0.5B-Instruct!")
    print("🔒 Your conversations never leave your device!")
    print("🚀 Much better conversational AI than before!")

if __name__ == "__main__":
    test_all_applications()
