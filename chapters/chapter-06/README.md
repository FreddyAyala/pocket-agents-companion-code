# Chapter 6: Pruning, Sparsity, and Distillation

**Pocket Agents: A Practical Guide to On-Device Artificial Intelligence**

This companion code demonstrates advanced model compression techniques including pruning, sparsity, and knowledge distillation to create smaller, faster models.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to this directory
cd companion-code/chapters/chapter-06

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
jupyter notebook pruning_sparsity_demo.ipynb
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

- **Magnitude-based pruning** removes weights with smallest absolute values
- **Sparsity patterns** in neural networks and their impact
- **Knowledge distillation** trains smaller models to mimic larger ones
- **Real-world deployment** scenarios and trade-offs
- **Performance vs quality** analysis for compression techniques

## 🎯 Key Concepts

- **Pruning**: Removing unnecessary weights (90% sparsity achievable)
- **Sparsity**: Percentage of zero weights in the model
- **Knowledge Distillation**: Teacher-student training paradigm
- **Compression Ratio**: Size reduction compared to original model
- **Quality Trade-offs**: Accuracy loss vs performance gains

## 📊 Expected Results

- **Pruning**: 90% sparsity with 2-5x speedup
- **Distillation**: 3-10x smaller models
- **Combined**: 5-20x overall improvement
- **Quality Loss**: 1-10% depending on technique

## 🔬 Techniques Demonstrated

### 1. Magnitude-Based Pruning
- L1 unstructured pruning
- Sparsity calculation
- Performance impact analysis

### 2. Knowledge Distillation
- Teacher-student model architecture
- Distillation loss function
- Training simulation

### 3. Real-World Analysis
- Deployment scenario feasibility
- Memory usage calculations
- Performance trade-offs

## 🚀 Performance Benefits

- **Faster Inference**: 2-5x speedup with pruning
- **Smaller Models**: 3-10x size reduction with distillation
- **Better Deployment**: Fits in mobile/edge constraints
- **Lower Power**: Reduced memory bandwidth requirements

## ⚖️ Trade-offs

- **Accuracy Loss**: 1-10% depending on compression level
- **Training Complexity**: Requires careful tuning
- **Hardware Dependencies**: Some techniques work better on specific hardware
- **Validation Required**: Must test on representative data

## 🔗 Related Chapters

- Chapter 4: Model Compression - Quantization
- Chapter 6: The Engines That Power Intelligence
- Chapter 7: The Hardware Battlefield
- Chapter 9: Taming the Edge: Memory, Context, and Concurrency

## 💡 Best Practices

1. **Start Simple**: Begin with magnitude-based pruning
2. **Validate Quality**: Test on representative data
3. **Combine Techniques**: Use pruning + distillation for maximum compression
4. **Hardware-Specific**: Optimize for target deployment platform
5. **Iterative Approach**: Gradually increase compression while monitoring quality
