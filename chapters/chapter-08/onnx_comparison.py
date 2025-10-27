"""
Chapter 8: ONNX Comparison - Cross-Platform Deployment

This module demonstrates ONNX Runtime as an alternative to GGUF/llama.cpp,
showing when and why you might choose each approach.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: transformers and/or torch not installed")
    print("Install with: pip install transformers torch")
    exit(1)

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Error: onnx and/or onnxruntime not installed")
    print("Install with: pip install onnx onnxruntime")
    exit(1)


def convert_pytorch_to_onnx(
    model_name: str,
    output_path: str,
    seq_length: int = 128
) -> str:
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model_name: HuggingFace model name or path
        output_path: Path to save ONNX model
        seq_length: Sequence length for dummy input
        
    Returns:
        Path to exported ONNX model
    """
    print("=" * 70)
    print("Converting PyTorch Model to ONNX")
    print("=" * 70)
    
    print(f"\nLoading model: {model_name}")
    
    # Load model (use a small model for demonstration)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, seq_length))
    
    print(f"Exporting to ONNX format...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Use recent opset version
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Check file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"✓ Model exported successfully")
    print(f"  Output: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print("=" * 70)
    
    return output_path


def get_available_providers() -> List[str]:
    """Get list of available ONNX Runtime execution providers."""
    providers = ort.get_available_providers()
    
    print("\nAvailable ONNX Runtime Providers:")
    print("-" * 70)
    for provider in providers:
        print(f"  • {provider}")
    print()
    
    return providers


def benchmark_onnx_inference(
    onnx_path: str,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference.
    
    Args:
        onnx_path: Path to ONNX model
        num_runs: Number of inference runs for averaging
        
    Returns:
        Dictionary with performance metrics
    """
    print("=" * 70)
    print("Benchmarking ONNX Inference")
    print("=" * 70)
    
    # Select best available provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    
    # Filter to only available providers
    providers = [p for p in providers if p in available]
    
    print(f"\nUsing providers: {providers}")
    
    # Create session with optimizations
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if 'CPUExecutionProvider' in providers:
        session_options.intra_op_num_threads = 8
        session_options.inter_op_num_threads = 2
    
    session = ort.InferenceSession(
        onnx_path,
        session_options,
        providers=providers
    )
    
    # Create test input
    test_input = np.random.randint(0, 1000, (1, 128), dtype=np.int64)
    
    # Warmup
    print(f"Warming up with 10 runs...")
    for _ in range(10):
        _ = session.run(None, {'input_ids': test_input})
    
    # Benchmark
    print(f"Running {num_runs} inference iterations...")
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        outputs = session.run(None, {'input_ids': test_input})
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults (averaged over {num_runs} runs):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print("=" * 70)
    
    return {
        'avg_ms': avg_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time
    }


def compare_gguf_vs_onnx():
    """
    Display comparison table between GGUF/llama.cpp and ONNX.
    """
    print("\n" + "=" * 70)
    print("GGUF/llama.cpp vs ONNX Runtime: When to Use Each")
    print("=" * 70)
    
    comparison = """
╔════════════════════╦═══════════════════════╦════════════════════════╗
║ Feature            ║ GGUF/llama.cpp        ║ ONNX Runtime           ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Best For           ║ Llama-family models   ║ Universal model types  ║
║                    ║ Maximum performance   ║ Enterprise deployment  ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Quantization       ║ Native 4-bit support  ║ Limited quantization   ║
║                    ║ Optimized kernels     ║ Requires external tools║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Memory Mapping     ║ Built-in, zero        ║ Standard file loading  ║
║                    ║ overhead loading      ║                        ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Model Support      ║ Llama, Mistral, Qwen  ║ Any PyTorch/TensorFlow ║
║                    ║ architectures         ║ model                  ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Deployment         ║ Desktop, mobile, web  ║ Server, cloud, edge    ║
║                    ║ (WASM)                ║ devices                ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Sovereignty        ║ 100% open-source      ║ Open standard          ║
║                    ║ Community-driven      ║ Microsoft-backed       ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Performance        ║ Excellent for LLMs    ║ Good for various       ║
║                    ║ Specialized kernels   ║ model types            ║
╠════════════════════╬═══════════════════════╬════════════════════════╣
║ Ease of Use        ║ Simple Python API     ║ Requires conversion    ║
║                    ║ Direct GGUF loading   ║ More setup             ║
╚════════════════════╩═══════════════════════╩════════════════════════╝
"""
    
    print(comparison)
    
    print("\n📊 Use Case Recommendations:")
    print("-" * 70)
    print("\nChoose GGUF/llama.cpp when:")
    print("  ✓ Working with Llama-family models (Llama, Mistral, Qwen)")
    print("  ✓ Need maximum on-device performance")
    print("  ✓ Want instant model loading (memory mapping)")
    print("  ✓ Building for mobile or web deployment")
    print("  ✓ Prioritize complete sovereignty and auditability")
    
    print("\nChoose ONNX Runtime when:")
    print("  ✓ Need to support multiple model architectures")
    print("  ✓ Working in enterprise with varying hardware")
    print("  ✓ Want vendor-neutral deployment standard")
    print("  ✓ Have existing ONNX conversion pipeline")
    print("  ✓ Need Microsoft's enterprise support")
    
    print("\n" + "=" * 70)


def demonstrate_onnx_optimization():
    """Demonstrate ONNX Runtime optimization options."""
    print("\n" + "=" * 70)
    print("ONNX Runtime Optimization Options")
    print("=" * 70)
    
    print("\nSession Optimization Levels:")
    print("-" * 70)
    
    optimization_levels = {
        'ORT_DISABLE_ALL': 'No optimizations',
        'ORT_ENABLE_BASIC': 'Basic graph optimizations (constant folding, etc.)',
        'ORT_ENABLE_EXTENDED': 'Extended optimizations (operator fusion)',
        'ORT_ENABLE_ALL': 'All optimizations (layout transformations, etc.)'
    }
    
    for level, description in optimization_levels.items():
        print(f"  • {level}")
        print(f"    {description}\n")
    
    print("Execution Providers by Hardware:")
    print("-" * 70)
    
    providers_info = {
        'CUDAExecutionProvider': 'NVIDIA GPUs (CUDA)',
        'TensorrtExecutionProvider': 'NVIDIA GPUs (TensorRT, optimized)',
        'ROCMExecutionProvider': 'AMD GPUs',
        'CoreMLExecutionProvider': 'Apple devices (Neural Engine)',
        'DmlExecutionProvider': 'DirectML (Windows, any GPU)',
        'OpenVINOExecutionProvider': 'Intel CPUs/GPUs',
        'CPUExecutionProvider': 'CPU (fallback, always available)'
    }
    
    for provider, description in providers_info.items():
        print(f"  • {provider}")
        print(f"    {description}\n")
    
    print("=" * 70)


def main():
    """Main demonstration function."""
    print("\n" + "=" * 70)
    print("Chapter 8: ONNX Comparison Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Show available providers
    get_available_providers()
    
    # Show comparison
    compare_gguf_vs_onnx()
    
    # Show optimization options
    demonstrate_onnx_optimization()
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("""
For the Hero Project, we chose GGUF/llama.cpp because:

1. Optimal for Llama-family models (our Qwen3-4B base)
2. Native 4-bit quantization support (Chapter 5)
3. Memory-mapped loading for instant startup
4. Complete open-source sovereignty
5. Excellent performance on consumer hardware

ONNX is a powerful alternative when you need:
- Universal model support across architectures
- Enterprise-grade vendor-neutral deployment
- Integration with existing Microsoft/.NET infrastructure

Both are excellent tools. The right choice depends on your specific
requirements, hardware, and deployment constraints.
""")
    
    print("=" * 70)
    print("\nNote: To run actual conversion and benchmarking:")
    print("  1. Uncomment the conversion code in this file")
    print("  2. Provide a small model name (e.g., 'distilbert-base-uncased')")
    print("  3. Ensure sufficient disk space for model files")
    print("=" * 70)


if __name__ == "__main__":
    main()

