#!/usr/bin/env python3
"""
Chapter 8: Test Setup Script

This script verifies that all dependencies are installed correctly
and all companion code files are working.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import sys
import importlib

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False

def main():
    """Test all dependencies and companion code."""
    print("=" * 70)
    print("Chapter 8: Testing Setup and Dependencies")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Test core dependencies
    print("\nTesting Core Dependencies:")
    print("-" * 70)
    
    dependencies = [
        ("torch", "PyTorch for model operations"),
        ("transformers", "HuggingFace transformers"),
        ("numpy", "Numerical operations"),
        ("llama_cpp", "llama-cpp-python for GGUF models"),
        ("onnx", "ONNX format support"),
        ("onnxruntime", "ONNX Runtime for inference"),
        ("jupyter", "Jupyter notebook support"),
    ]
    
    all_passed = True
    for module, description in dependencies:
        if not test_import(module, description):
            all_passed = False
    
    # Test companion code files
    print("\nTesting Companion Code Files:")
    print("-" * 70)
    
    try:
        import architectural_optimizations
        print("✓ architectural_optimizations.py - Can be imported")
    except Exception as e:
        print(f"❌ architectural_optimizations.py: {e}")
        all_passed = False
    
    try:
        import hero_project_engine_demo
        print("✓ hero_project_engine_demo.py - Can be imported")
    except Exception as e:
        print(f"❌ hero_project_engine_demo.py: {e}")
        all_passed = False
    
    try:
        import onnx_comparison
        print("✓ onnx_comparison.py - Can be imported")
    except Exception as e:
        print(f"❌ onnx_comparison.py: {e}")
        all_passed = False
    
    # Test specific functionality
    print("\nTesting Specific Functionality:")
    print("-" * 70)
    
    try:
        from architectural_optimizations import OptimizedKVCache
        cache = OptimizedKVCache(1024, 8, 64)
        print("✓ KV Cache - Can create and use cache")
    except Exception as e:
        print(f"❌ KV Cache: {e}")
        all_passed = False
    
    try:
        from onnx_comparison import get_available_providers
        providers = get_available_providers()
        print(f"✓ ONNX Runtime - Found {len(providers)} providers")
    except Exception as e:
        print(f"❌ ONNX Runtime: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Chapter 8 companion code is ready.")
        print("\nNext steps:")
        print("  1. Run: python architectural_optimizations.py")
        print("  2. Run: python onnx_comparison.py")
        print("  3. Run: jupyter notebook inference_engines_demo.ipynb")
        print("  4. For Hero Project: Download model and run hero_project_engine_demo.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo fix:")
        print("  1. Run: pip install -r requirements.txt")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Ensure virtual environment is activated")
    
    print("=" * 70)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
