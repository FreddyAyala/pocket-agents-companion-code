#!/usr/bin/env python3
"""
Chapter 10: Universal Deployment with llama.cpp

This script demonstrates universal deployment using llama.cpp,
which runs efficiently across all platforms without external dependencies.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import time
import psutil
import os
from pathlib import Path

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠️ llama-cpp-python not available. Install with: pip install llama-cpp-python")

class UniversalDeployment:
    """Universal deployment using llama.cpp for cross-platform AI"""
    
    def __init__(self, model_path=None, **kwargs):
        """Initialize universal deployment engine"""
        
        if not LLAMA_AVAILABLE:
            raise ImportError("llama-cpp-python is required for universal deployment")
        
        # Default parameters optimized for on-device deployment
        default_params = {
            'n_ctx': 2048,           # Context window
            'n_batch': 512,          # Batch size
            'n_threads': 4,          # CPU threads
            'n_gpu_layers': 0,       # GPU layers (0 = CPU only)
            'use_mmap': True,        # Memory mapping
            'use_mlock': False,      # Lock memory
            'verbose': False         # Quiet mode
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        
        # Use demo model if no path provided
        if model_path is None:
            model_path = self._get_demo_model()
        
        print(f"🚀 Loading universal deployment model...")
        print(f"   Model: {model_path}")
        print(f"   Context: {params['n_ctx']}")
        print(f"   Threads: {params['n_threads']}")
        
        # Initialize model
        self.llm = Llama(model_path=model_path, **params)
        
        # Performance tracking
        self.inference_times = []
        self.token_counts = []
        self.memory_usage = []
        
        print("✅ Universal deployment engine ready!")
    
    def _get_demo_model(self):
        """Get demo model path or provide instructions"""
        # In a real scenario, you would download a GGUF model
        # For this demo, we'll show the expected structure
        demo_models = [
            "models/llama-3.1-7b-instruct-q4_k_m.gguf",
            "models/mistral-7b-instruct-v0.2-q4_k_m.gguf",
            "models/qwen3-4b-instruct-q4_k_m.gguf"
        ]
        
        for model_path in demo_models:
            if os.path.exists(model_path):
                return model_path
        
        # If no model found, provide instructions
        print("📥 No GGUF model found. To use this demo:")
        print("   1. Download a GGUF model from HuggingFace:")
        print("      - TheBloke/Llama-3.1-7B-Instruct-GGUF")
        print("      - TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        print("      - Qwen/Qwen3-4B-Instruct-GGUF")
        print("   2. Place the .gguf file in the models/ directory")
        print("   3. Update the model_path in the script")
        print("")
        print("For now, using mock responses...")
        return None
    
    def generate(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
        """Generate text with performance tracking"""
        
        if self.llm is None:
            return self._mock_generate(prompt, max_tokens)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "\n\n"],
            echo=False,
            **kwargs
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Track performance
        inference_time = end_time - start_time
        token_count = len(response['choices'][0]['text'].split())
        memory_used = end_memory - start_memory
        
        self.inference_times.append(inference_time)
        self.token_counts.append(token_count)
        self.memory_usage.append(memory_used)
        
        return response
    
    def _mock_generate(self, prompt, max_tokens):
        """Mock generation for demo purposes"""
        time.sleep(0.1)  # Simulate inference time
        
        mock_response = f"Mock response to: '{prompt[:50]}...' (This is a demo response. Install a real GGUF model for actual inference.)"
        
        return {
            'choices': [{'text': mock_response}],
            'usage': {'total_tokens': len(mock_response.split())}
        }
    
    def get_performance_stats(self):
        """Get performance statistics"""
        
        if not self.inference_times:
            return None
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        avg_tokens = sum(self.token_counts) / len(self.token_counts)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        throughput = avg_tokens / avg_time
        
        return {
            'average_inference_time': avg_time,
            'average_tokens': avg_tokens,
            'average_memory_usage': avg_memory,
            'throughput_tokens_per_second': throughput,
            'total_inferences': len(self.inference_times)
        }
    
    def optimize_for_platform(self, platform):
        """Optimize llama.cpp for specific platform"""
        
        print(f"🔧 Optimizing for {platform} platform...")
        
        if platform == 'mobile':
            # Mobile optimizations
            self.llm.params.n_ctx = 1024      # Smaller context
            self.llm.params.n_batch = 256     # Smaller batch
            self.llm.params.n_threads = 2      # Fewer threads
            self.llm.params.use_mmap = True   # Memory mapping
            print("   ✅ Mobile optimizations applied")
            
        elif platform == 'desktop':
            # Desktop optimizations
            self.llm.params.n_ctx = 4096      # Larger context
            self.llm.params.n_batch = 512      # Larger batch
            self.llm.params.n_threads = 8     # More threads
            self.llm.params.use_mmap = True   # Memory mapping
            print("   ✅ Desktop optimizations applied")
            
        elif platform == 'server':
            # Server optimizations
            self.llm.params.n_ctx = 8192      # Large context
            self.llm.params.n_batch = 1024    # Large batch
            self.llm.params.n_threads = 16    # Many threads
            self.llm.params.use_mmap = True   # Memory mapping
            print("   ✅ Server optimizations applied")
        
        else:
            print(f"   ⚠️ Unknown platform: {platform}")

def demonstrate_universal_deployment():
    """Demonstrate universal deployment capabilities"""
    
    print("=" * 70)
    print("Chapter 10: Universal Deployment Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    try:
        # Initialize deployment engine
        deployment = UniversalDeployment()
        
        # Test generation
        print("\n🧪 Testing universal deployment...")
        test_prompts = [
            "Hello! How are you?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about AI."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            response = deployment.generate(prompt, max_tokens=50)
            print(f"🤖 Response: {response['choices'][0]['text']}")
        
        # Show performance stats
        stats = deployment.get_performance_stats()
        if stats:
            print(f"\n📊 Performance Statistics:")
            print(f"   Average inference time: {stats['average_inference_time']:.3f}s")
            print(f"   Average tokens: {stats['average_tokens']:.1f}")
            print(f"   Throughput: {stats['throughput_tokens_per_second']:.1f} tokens/sec")
            print(f"   Total inferences: {stats['total_inferences']}")
        
        # Demonstrate platform optimization
        print(f"\n🔧 Platform Optimization Demo:")
        for platform in ['mobile', 'desktop', 'server']:
            deployment.optimize_for_platform(platform)
        
        print(f"\n✅ Universal deployment demonstration complete!")
        print(f"💡 Key advantages:")
        print(f"   • Zero external dependencies")
        print(f"   • Cross-platform compatibility")
        print(f"   • Memory-mapped loading")
        print(f"   • Hardware-optimized kernels")
        
    except Exception as e:
        print(f"❌ Error in universal deployment demo: {e}")
        print(f"💡 Make sure llama-cpp-python is installed: pip install llama-cpp-python")

if __name__ == "__main__":
    demonstrate_universal_deployment()
