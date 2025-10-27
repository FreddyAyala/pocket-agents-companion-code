#!/usr/bin/env python3
"""
Test script for Chapter 7: Enhanced GPU Backend Support
Demonstrates how to test GPU performance even when CUDA is not available
"""

import sys
import time

def test_gpu_backends():
    """Test available GPU backends"""
    
    print("🔍 Testing GPU Backend Support")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return
    
    backends = []
    
    # Check CUDA
    if torch.cuda.is_available():
        backends.append({
            'name': 'CUDA (NVIDIA)',
            'device': 'cuda',
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
        print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i, name in enumerate(backends[-1]['gpu_names']):
            print(f"   GPU {i}: {name}")
    else:
        print("❌ CUDA not available")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        backends.append({
            'name': 'MPS (Apple Silicon)',
            'device': 'mps',
            'gpu_count': 1,
            'gpu_names': ['Apple Silicon GPU']
        })
        print("✅ MPS (Apple Silicon) available")
    else:
        print("❌ MPS (Apple Silicon) not available")
    
    # Check Intel XPU
    try:
        import torch.xpu
        if torch.xpu.is_available():
            backends.append({
                'name': 'Intel XPU',
                'device': 'xpu',
                'gpu_count': torch.xpu.device_count(),
                'gpu_names': ['Intel GPU']
            })
            print("✅ Intel XPU available")
        else:
            print("❌ Intel XPU not available")
    except ImportError:
        print("❌ Intel XPU not installed")
    
    # Check ROCm (AMD)
    try:
        if torch.version.hip is not None:
            backends.append({
                'name': 'ROCm (AMD)',
                'device': 'cuda',  # ROCm uses CUDA API
                'gpu_count': 1,
                'gpu_names': ['AMD GPU']
            })
            print("✅ ROCm (AMD) available")
        else:
            print("❌ ROCm (AMD) not available")
    except:
        print("❌ ROCm (AMD) not available")
    
    print(f"\n📊 Summary: {len(backends)} GPU backend(s) available")
    
    if backends:
        print("\n🎯 Available GPU backends for testing:")
        for backend in backends:
            print(f"   • {backend['name']}: {backend['gpu_count']} GPU(s)")
            for gpu_name in backend['gpu_names']:
                print(f"     - {gpu_name}")
        
        # Test the first available backend
        backend = backends[0]
        print(f"\n🧪 Testing {backend['name']}...")
        
        try:
            # Create test tensor
            test_tensor = torch.randn(100, 100)
            test_tensor_gpu = test_tensor.to(backend['device'])
            
            # Simple computation test
            start_time = time.time()
            result = torch.matmul(test_tensor_gpu, test_tensor_gpu.T)
            
            # Synchronize based on backend
            if backend['device'] == 'cuda':
                torch.cuda.synchronize()
            elif backend['device'] == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            
            print(f"   ✅ {backend['name']} test successful")
            print(f"   ⏱️ Computation time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"   📊 Result shape: {result.shape}")
            
        except Exception as e:
            print(f"   ❌ {backend['name']} test failed: {e}")
    
    else:
        print("\nℹ️ No GPU backends available - CPU-only testing")
        print("   This is normal on systems without GPU support")
        print("   The notebook will still demonstrate CPU performance analysis")
    
    return backends

if __name__ == "__main__":
    test_gpu_backends()
