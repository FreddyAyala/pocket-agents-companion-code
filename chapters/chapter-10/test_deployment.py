#!/usr/bin/env python3
"""
Chapter 10: Deployment Testing Script

This script tests all deployment strategies and provides a comprehensive
overview of the deployment capabilities.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import sys
import time
import subprocess
import threading
from pathlib import Path

def test_imports():
    """Test all required imports"""
    
    print("🧪 Testing imports...")
    print("-" * 50)
    
    imports = [
        ("torch", "PyTorch for model operations"),
        ("transformers", "HuggingFace transformers"),
        ("gradio", "Gradio for web interfaces"),
        ("streamlit", "Streamlit for web apps"),
        ("fastapi", "FastAPI for production APIs"),
        ("uvicorn", "ASGI server for FastAPI"),
        ("psutil", "System monitoring"),
        ("pydantic", "Data validation"),
    ]
    
    all_passed = True
    for module, description in imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            all_passed = False
    
    return all_passed

def test_universal_deployment():
    """Test universal deployment"""
    
    print("\n🌐 Testing Universal Deployment...")
    print("-" * 50)
    
    try:
        from universal_deployment import UniversalDeployment, demonstrate_universal_deployment
        
        print("✅ Universal deployment module imported successfully")
        
        # Test demonstration
        demonstrate_universal_deployment()
        
        return True
        
    except Exception as e:
        print(f"❌ Universal deployment test failed: {e}")
        return False

def test_web_deployment():
    """Test web deployment"""
    
    print("\n🌐 Testing Web Deployment...")
    print("-" * 50)
    
    try:
        from web_deployment import WebDeployment, demonstrate_web_deployment
        
        print("✅ Web deployment module imported successfully")
        
        # Test demonstration
        demonstrate_web_deployment()
        
        return True
        
    except Exception as e:
        print(f"❌ Web deployment test failed: {e}")
        return False

def test_production_deployment():
    """Test production deployment"""
    
    print("\n🏭 Testing Production Deployment...")
    print("-" * 50)
    
    try:
        from production_deployment import ProductionAIEngine, demonstrate_production_deployment
        
        print("✅ Production deployment module imported successfully")
        
        # Test demonstration
        demonstrate_production_deployment()
        
        return True
        
    except Exception as e:
        print(f"❌ Production deployment test failed: {e}")
        return False

def test_jupyter_notebook():
    """Test Jupyter notebook"""
    
    print("\n📓 Testing Jupyter Notebook...")
    print("-" * 50)
    
    notebook_path = Path("deployment_playbook_demo.ipynb")
    
    if not notebook_path.exists():
        print("❌ Jupyter notebook not found")
        return False
    
    try:
        import nbformat
        from nbconvert import PythonExporter
        
        # Load and validate notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        print(f"✅ Jupyter notebook found: {notebook_path}")
        print(f"   Cells: {len(notebook.cells)}")
        
        # Check for code cells
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        print(f"   Code cells: {len(code_cells)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Jupyter notebook test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive deployment test"""
    
    print("=" * 70)
    print("Chapter 10: Comprehensive Deployment Test")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    tests = [
        ("Import Test", test_imports),
        ("Universal Deployment", test_universal_deployment),
        ("Web Deployment", test_web_deployment),
        ("Production Deployment", test_production_deployment),
        ("Jupyter Notebook", test_jupyter_notebook),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Chapter 10 deployment code is ready.")
        print("\n💡 Next steps:")
        print("   1. Run: python universal_deployment.py")
        print("   2. Run: python web_deployment.py")
        print("   3. Run: python production_deployment.py server")
        print("   4. Run: jupyter notebook deployment_playbook_demo.ipynb")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
        print("\n💡 To fix issues:")
        print("   1. Run: pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Ensure virtual environment is activated")
    
    return passed == total

def show_deployment_options():
    """Show available deployment options"""
    
    print("\n" + "=" * 70)
    print("🚀 Available Deployment Options")
    print("=" * 70)
    
    print("\n1. 🌐 Universal Deployment (llama.cpp)")
    print("   • Cross-platform compatibility")
    print("   • Zero external dependencies")
    print("   • Memory-mapped loading")
    print("   • Command: python universal_deployment.py")
    
    print("\n2. 🌐 Web Deployment (Gradio + Streamlit)")
    print("   • Interactive web interfaces")
    print("   • Real-time chat experience")
    print("   • Easy sharing and deployment")
    print("   • Command: python web_deployment.py launch")
    
    print("\n3. 🏭 Production Deployment (FastAPI)")
    print("   • Production-ready API")
    print("   • Comprehensive monitoring")
    print("   • Error handling and recovery")
    print("   • Command: python production_deployment.py server")
    
    print("\n4. 📓 Interactive Learning (Jupyter)")
    print("   • Step-by-step tutorials")
    print("   • Hands-on experimentation")
    print("   • Visual demonstrations")
    print("   • Command: jupyter notebook deployment_playbook_demo.ipynb")
    
    print("\n💡 Choose the deployment strategy that best fits your needs!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "options":
        show_deployment_options()
    else:
        success = run_comprehensive_test()
        if success:
            show_deployment_options()
        sys.exit(0 if success else 1)
