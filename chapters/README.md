# Companion Code for "On-Device AI: The Small Language Models Revolution"

## Overview

This directory contains practical code examples and implementations for each chapter of the book. The code is organized by chapter and designed to be runnable, testable, and educational.

## Structure

```
chapters/
├── chapter-04/          # Model Compression & Quantization
├── chapter-05/          # Pruning, Sparsity & Distillation  
├── chapter-06/          # Inference Engines
├── chapter-09/          # Edge Management
├── chapter-10/          # RAG On-Device
├── chapter-11/          # Agentic Workflows
└── chapter-12/          # Hero Project (Complete System)
```

## Running the Examples

### Prerequisites

```bash
# Install required packages
pip install torch numpy psutil

# For advanced examples (Chapters 10-12)
pip install llama-cpp-python sentence-transformers fastapi uvicorn
```

### Quick Start

```bash
# Run Chapter 4 quantization example
cd chapter-04
python quantization_example.py

# Run Chapter 10 RAG example
cd chapter-10
python rag_demo.py

# Run Chapter 12 Hero Project
cd chapter-12
python main.py
```

## Chapter-Specific Examples

### Chapter 4: Model Compression
- **quantization_example.py**: Demonstrates FP32 → INT8 quantization
- **gguf_conversion.py**: Shows GGUF format conversion
- **performance_comparison.py**: Benchmarks different quantization levels

### Chapter 5: Pruning & Distillation
- **pruning_demo.py**: Structured vs unstructured pruning
- **distillation_example.py**: Teacher-student knowledge transfer
- **sparsity_analysis.py**: Analyzes model sparsity patterns

### Chapter 6: Inference Engines
- **llama_cpp_demo.py**: llama.cpp integration example
- **onnx_runtime_demo.py**: ONNX Runtime optimization
- **engine_comparison.py**: Performance comparison across engines

### Chapter 9: Edge Management
- **memory_manager.py**: Dynamic memory allocation
- **context_manager.py**: Context window optimization
- **concurrency_demo.py**: Parallel processing examples

### Chapter 10: RAG On-Device
- **vector_database.py**: SQLite-based vector store
- **rag_pipeline.py**: Complete RAG implementation
- **hybrid_search.py**: Vector + keyword search

### Chapter 11: Agentic Workflows
- **function_calling.py**: Local function execution
- **agent_loop.py**: Agent execution framework
- **multimodal_demo.py**: Vision + voice integration

### Chapter 12: Hero Project
- **main.py**: Complete Private Local Assistant
- **web_interface.py**: FastAPI web interface
- **mobile_app.py**: React Native mobile app

## Testing

Each chapter includes unit tests:

```bash
# Run all tests
python -m pytest

# Run specific chapter tests
python -m pytest chapter-04/tests/
```

## Contributing

When adding new examples:

1. **Keep it simple**: Focus on core concepts, not complex implementations
2. **Make it runnable**: Include all dependencies and setup instructions
3. **Add documentation**: Explain what the code demonstrates
4. **Include tests**: Ensure code works as expected
5. **Follow the book**: Align with chapter content and narrative

## License

This companion code is released under the same license as the book.
