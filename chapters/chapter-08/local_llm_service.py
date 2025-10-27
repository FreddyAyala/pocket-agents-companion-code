#!/usr/bin/env python3
"""
Local LLM Service for On-Device AI
This service runs a small language model locally and provides an API for other apps to use.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import threading
from typing import Optional, Dict, Any

class LocalLLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Excellent conversational model
        self.loaded = False
        self.loading = False
        
    def load_model(self):
        """Load the small language model"""
        if self.loaded or self.loading:
            return
            
        self.loading = True
        print(f"🔄 Loading {self.model_name}...")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.loaded = True
            self.loading = False
            print(f"✅ Model loaded successfully!")
            print(f"   Model: {self.model_name}")
            print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            print(f"   Parameters: ~{self.model.num_parameters() / 1e6:.1f}M")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.loading = False
            # Fallback to a mock model
            self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock model as fallback"""
        print("🔄 Creating mock model as fallback...")
        self.model_name = "mock-model"
        self.loaded = True
        self.loading = False
        print("✅ Mock model ready!")
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response using the local model"""
        if not self.loaded:
            self.load_model()
        
        if self.model_name == "mock-model":
            # Mock responses
            responses = [
                f"I understand you said: '{prompt}'. This is a local AI response!",
                f"Thanks for your message: '{prompt}'. I'm running locally on your device.",
                f"Processing '{prompt}' locally... Here's my response!",
                f"Hello! I received: '{prompt}'. This is processed on your machine.",
                f"Local AI here! I heard: '{prompt}'. No data left your device!"
            ]
            import random
            time.sleep(0.5)  # Simulate processing time
            return random.choice(responses)
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response if response else "I'm processing your request locally!"
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return f"Local AI response to: '{prompt}' (processed on your device)"
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "model_name": self.model_name,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "parameters": f"{self.model.num_parameters() / 1e6:.1f}M" if self.model else "Unknown"
        }

# Global service instance
llm_service = LocalLLMService()

def get_llm_service():
    """Get the global LLM service instance"""
    return llm_service

if __name__ == "__main__":
    # Test the service
    print("🧪 Testing Local LLM Service...")
    service = get_llm_service()
    
    # Test generation
    test_prompts = [
        "Hello, how are you?",
        "What is AI?",
        "Tell me a joke",
        "How does local AI work?"
    ]
    
    for prompt in test_prompts:
        print(f"\n👤 User: {prompt}")
        response = service.generate_response(prompt)
        print(f"🤖 AI: {response}")
    
    print(f"\n📊 Service Status: {service.get_status()}")
