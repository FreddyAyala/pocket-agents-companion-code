# Chapter 5: Model Compression - Quantization

**Pocket Agents: A Practical Guide to On-Device Artificial Intelligence**

This companion code demonstrates model compression techniques, focusing on quantization to reduce model size and improve inference speed while maintaining performance.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to this directory
cd companion-code/chapters/chapter-05

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

# 3. Install Jupyter kernel
python -m ipykernel install --user --name=venv --display-name="Python (venv)"

# 4. Launch Jupyter
jupyter notebook quantization_demo.ipynb
```

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

This happens when Jupyter is not using the virtual environment. **Solution:**

1. **Check the kernel** in the top-right corner of Jupyter
2. **Select "Python (venv)"** kernel (not the default Python kernel)
3. **If not available**, run:
   ```bash
   source venv/bin/activate
   python -m ipykernel install --user --name=venv --display-name="Python (venv)"
   ```
4. **Restart Jupyter** and select the correct kernel

### Other Common Issues

- **"Permission denied"**: Run `chmod +x setup_and_test.sh` first
- **"Python not found"**: Install Python 3.8+ from python.org
- **"Jupyter not launching"**: Try `jupyter lab` instead

## 📋 What You'll Learn

- How quantization reduces model size by 75%
- The trade-offs between precision and performance
- Real-world impact on mobile vs desktop deployment
- Hands-on quantization of neural network models
- Visualization of quantization effects

## 🎯 Key Concepts

- **Quantization**: Reducing precision from 32-bit to 8-bit or 4-bit
- **Memory Reduction**: 4x smaller with INT8, 8x smaller with INT4
- **Performance**: 2-4x speedup, 50-75% power reduction
- **Quality**: Minimal accuracy loss (usually <5%)

## 📊 Expected Results

- Original model: ~22 KB (FP32)
- Quantized model: ~5.5 KB (INT8)
- Size reduction: 75%
- Quality retention: 95%+

## 🔗 Related Chapters

- Chapter 5: Pruning, Sparsity, and Distillation
- Chapter 6: Inference Engines
- Chapter 7: Hardware Optimization
