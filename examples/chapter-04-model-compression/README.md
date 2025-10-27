# Chapter 4: Model Compression Examples

## Overview
This directory contains code examples for Chapter 4: "Model Compression: The Art of Quantization" from "On-Device AI: The Small Language Models Revolution."

## Examples Included

### 1. Basic Quantization (`basic_quantization.py`)
- Demonstrates FP32 to INT8 quantization
- Shows memory usage before and after quantization
- Includes performance comparison

### 2. Advanced Quantization (`advanced_quantization.py`)
- GPTQ quantization implementation
- Group-wise quantization techniques
- Quality vs. compression trade-offs

### 3. GGUF Conversion (`gguf_conversion.py`)
- Converting models to GGUF format
- llama.cpp integration
- Cross-platform compatibility

### 4. Quantization Comparison (`quantization_comparison.py`)
- Side-by-side comparison of different quantization methods
- Performance benchmarks
- Quality assessment tools

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install torch transformers accelerate
   ```

2. **Run basic quantization example**:
   ```bash
   python basic_quantization.py
   ```

3. **Compare different methods**:
   ```bash
   python quantization_comparison.py
   ```

## Key Learning Objectives

- Understand different quantization techniques
- Learn to balance compression vs. quality
- Master GGUF format for cross-platform deployment
- Implement quantization in production systems

## Performance Targets

| Quantization Method | Compression Ratio | Quality Retention | Speed Improvement |
|-------------------|------------------|------------------|------------------|
| INT8 | 4x | 95% | 2x |
| INT4 | 8x | 90% | 3x |
| GPTQ | 4x | 98% | 2.5x |

## Files in this Directory

- `basic_quantization.py` - Basic quantization example
- `advanced_quantization.py` - Advanced quantization techniques
- `gguf_conversion.py` - GGUF format conversion
- `quantization_comparison.py` - Performance comparison
- `README.md` - This file

## Related Book Content

This code supports the following sections in Chapter 4:
- Quantization basics and bit depth reduction
- Advanced quantization methods (GPTQ)
- GGUF format and llama.cpp integration
- Performance optimization techniques

---

*These examples demonstrate the practical application of model compression techniques for sovereign AI applications.*
