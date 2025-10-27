#!/usr/bin/env python3
"""
Chapter 4: Model Compression - Quantization Example
Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence

This example demonstrates basic quantization concepts for on-device AI.
Run this to see quantization in action with a simple model.
"""

import torch
import torch.nn as nn
import numpy as np
import time

class SimpleModel(nn.Module):
    """A simple model to demonstrate quantization"""
    
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def demonstrate_quantization():
    """Demonstrate the impact of quantization on model size and performance"""
    
    print("🔧 Chapter 4: Model Compression - Quantization Demo")
    print("=" * 60)
    
    # Create a simple model
    model = SimpleModel()
    
    # Generate some test data
    input_data = torch.randn(1, 100)
    
    # Measure original model
    print("\n📊 Original Model (FP32):")
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"   Model size: {original_size / 1024:.2f} KB")
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    original_time = time.time() - start_time
    print(f"   Inference time: {original_time * 1000:.2f} ms")
    
    # Quantize to INT8
    print("\n⚡ Quantized Model (INT8):")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    print(f"   Model size: {quantized_size / 1024:.2f} KB")
    print(f"   Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
    
    # Time quantized inference
    start_time = time.time()
    with torch.no_grad():
        quantized_output = quantized_model(input_data)
    quantized_time = time.time() - start_time
    print(f"   Inference time: {quantized_time * 1000:.2f} ms")
    print(f"   Speed improvement: {original_time/quantized_time:.1f}x")
    
    # Check output similarity
    output_diff = torch.abs(output - quantized_output).mean()
    print(f"   Output difference: {output_diff:.6f}")
    
    print("\n✅ Key Takeaways:")
    print("   • Quantization reduces model size by ~75%")
    print("   • Inference speed improves significantly")
    print("   • Output quality remains nearly identical")
    print("   • Perfect for on-device deployment!")

if __name__ == "__main__":
    demonstrate_quantization()
