# Chapter 8: The Engines That Power Intelligence

**Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence**

This companion code demonstrates the various inference engines and runtimes that power on-device AI, showing how optimized models from Chapters 5-7 become production-ready systems.

## 📁 Contents

### Core Implementation Files

- **`architectural_optimizations.py`** - Complete implementations of KV Cache, GQA, and SWA
  - Optimized KV cache for autoregressive generation
  - Grouped Query Attention (GQA) with memory savings
  - Sliding Window Attention (SWA) for infinite context
  - Performance benchmarking functions

- **`hero_project_engine_demo.py`** - Production-ready Hero Project implementation
  - Complete model loading with error handling
  - Performance monitoring and metrics
  - Context management demonstrations
  - Manifesto compliance checking

- **`onnx_comparison.py`** - ONNX Runtime demonstrations and comparisons
  - PyTorch to ONNX conversion
  - ONNX Runtime inference
  - GGUF vs ONNX comparison tables
  - Provider optimization examples

- **`inference_engines_demo.ipynb`** - Interactive Jupyter notebook
  - All demonstrations in one place
  - Visual performance comparisons
  - Interactive experimentation

### Supporting Files

- **`requirements.txt`** - Python dependencies
- **`setup_and_test.sh`** - Setup script for dependencies
- **`README.md`** - This file

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Navigate to this directory
cd companion-code/chapters/chapter-08

# Run the setup script
./setup_and_test.sh
```

### Option 2: Manual Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run demonstrations
python architectural_optimizations.py
python hero_project_engine_demo.py
python onnx_comparison.py

# Or launch Jupyter notebook
jupyter notebook inference_engines_demo.ipynb
```

## 📚 What This Demonstrates

### 1. Architectural Innovations (architectural_optimizations.py)

Learn how Chapter 6's "slimming" techniques translate to real engine optimizations:

- **KV Cache**: Transform O(n²) to O(n) complexity
- **Grouped Query Attention (GQA)**: 4-8x memory reduction
- **Sliding Window Attention (SWA)**: Infinite context with linear cost

```bash
python architectural_optimizations.py
```

**Expected Output**: Benchmarking results showing memory savings and performance improvements.

### 2. Hero Project Integration (hero_project_engine_demo.py)

See the complete pipeline from Chapters 5-7 come together:

- **Chapter 5**: Quantized GGUF model (75% size reduction)
- **Chapter 7**: Fine-tuned for agentic tasks
- **Chapter 8**: Deployed with llama.cpp engine

```bash
# Set model path (or use default)
export HERO_MODEL_PATH="./models/qwen3-4b-q4_K_M.gguf"

python hero_project_engine_demo.py
```

**Expected Output**:
- ✓ TTFT < 150ms (Manifesto requirement)
- ✓ Throughput > 200 tok/s
- Complete performance analysis

### 3. ONNX Comparison (onnx_comparison.py)

Understand when to use GGUF vs ONNX:

- PyTorch to ONNX conversion
- Cross-platform deployment options
- Performance and sovereignty tradeoffs

```bash
python onnx_comparison.py
```

**Expected Output**: Comparison tables and use case recommendations.

## 🎯 Learning Objectives

After completing this chapter's companion code, you will understand:

1. **How engines transform optimized models into production systems**
   - Memory mapping for instant loading
   - Hardware-specific kernel optimization
   - KV cache management

2. **The tradeoffs between different inference engines**
   - GGUF/llama.cpp: Maximum on-device performance
   - ONNX: Universal cross-platform deployment
   - Proprietary solutions: Vendor-specific optimization

3. **How architectural innovations reduce computational cost**
   - KV cache: O(n²) → O(n) complexity
   - GQA: 4-8x memory reduction
   - SWA: Infinite context with linear cost

4. **How the Hero Project achieves Manifesto-compliant performance**
   - TTFT < 150ms
   - Throughput > 200 tok/s
   - Memory < 4GB

## 📊 Performance Benchmarks

### Hero Project Performance (Qwen3-4B Q4_K_M)

| Metric | Target | Achieved | Status |
|:---|:---:|:---:|:---:|
| TTFT | < 150ms | 80-120ms | ✓ |
| Throughput | > 200 tok/s | 200-300 tok/s | ✓ |
| Memory | < 4GB | 2.5-4GB | ✓ |

### Comparative Analysis

| Engine Type | TTFT (ms) | Throughput | Memory |
|:---|:---:|:---:|:---:|
| Cloud API | 200-500 | 50-100 | N/A |
| PyTorch (CPU) | 800-1200 | 10-20 | 14-16GB |
| **Hero Project** | **80-120** | **200-300** | **2.5-4GB** |

## 🔧 Troubleshooting

### Issue: llama-cpp-python not loading model

**Solution**:
```bash
# Ensure model file exists
ls -lh ./models/qwen3-4b-q4_K_M.gguf

# Check if llama-cpp-python is installed correctly
python -c "from llama_cpp import Llama; print('OK')"

# Reinstall with GPU support (optional)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall llama-cpp-python
```

### Issue: Out of memory

**Solution**:
```python
# Reduce context window or GPU layers
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Reduce from 4096
    n_gpu_layers=20,  # Reduce from -1 (all)
)
```

### Issue: ONNX conversion fails

**Solution**:
```bash
# Ensure compatible versions
pip install --upgrade onnx onnxruntime torch transformers

# Use smaller model for testing
# Replace large model with distilbert-base-uncased
```

## 🔗 Related Chapters

- **Chapter 5**: Model Compression (creates the GGUF files we load)
- **Chapter 6**: Pruning & Distillation (enables GQA/SWA architectures)
- **Chapter 7**: Fine-Tuning (specializes models for specific tasks)
- **Chapter 9**: Hardware Battlefield (next: hardware acceleration)
- **Chapter 10**: Deployment Playbook (next: cross-platform deployment)

## 📖 Additional Resources

### GGUF/llama.cpp
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

### ONNX Runtime
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

### Performance Optimization
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [Sliding Window Attention](https://arxiv.org/abs/2004.05150)
- [KV Cache Optimization](https://arxiv.org/abs/2211.05102)

## 💡 Next Steps

1. **Experiment with different models**
   - Try other Qwen models from HuggingFace
   - Compare performance across model sizes

2. **Optimize for your hardware**
   - Tune `n_gpu_layers` for your GPU
   - Experiment with `n_threads` for CPU performance

3. **Build your own agent**
   - Use Hero Project as a template
   - Integrate with your own tools and data

4. **Continue to Part III**
   - Chapter 9: Understanding hardware acceleration
   - Chapter 10: Deploying across platforms

---

**Need Help?** Check the troubleshooting section above or refer to the main book chapter for detailed explanations.
