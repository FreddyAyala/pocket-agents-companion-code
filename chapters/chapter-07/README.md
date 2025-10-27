# Chapter 7: Fine-Tuning & Adaptation

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to this directory
cd companion-code/chapters/chapter-07

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
python -m ipykernel install --user --name=chapter07 --display-name="Python (chapter07)"

# 4. Launch Jupyter
jupyter notebook lora_finetuning_demo.ipynb
```

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'peft'"

This happens when Jupyter is not using the virtual environment. **Solution:**

1. **Check the kernel** in the top-right corner of Jupyter
2. **Select "Python (chapter07)"** kernel (not the default Python kernel)
3. **If not available**, run this in terminal:
   ```bash
   source venv/bin/activate
   python -m ipykernel install --user --name=chapter07 --display-name="Python (chapter07)"
   ```
4. **Restart Jupyter** and select the correct kernel

### Other Common Issues

- **"Command not found"**: Make sure you're in the correct directory
- **"Permission denied"**: Run `chmod +x setup_and_test.sh` first
- **"Python not found"**: Install Python 3.8+ from python.org
- **"CUDA out of memory"**: Reduce batch size or use CPU training

## 📋 What You'll Learn

- **LoRA (Low-Rank Adaptation)** with TinyLlama-1.1B-Chat
- **Parameter efficiency** - training only ~0.5% of model parameters
- **Before/after comparison** showing clear improvement
- **Instruction-tuned models** vs generic models
- **On-device training** on consumer hardware (MPS/CPU)

## 🎯 Key Concepts

- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **QLoRA**: Quantized LoRA for memory-constrained environments
- **PEFT**: Parameter-efficient fine-tuning techniques
- **Domain Adaptation**: Customizing models for specific tasks
- **Gradient Accumulation**: Training with limited memory
- **Mixed Precision**: FP16/BF16 training for faster convergence

## 🚀 Expected Results

After running this demo, you'll have:

- **Working LoRA fine-tuning** in 3-5 minutes
- **Clear before/after comparison** showing improvement
- **Trained LoRA adapters** saved for deployment
- **Understanding** of why instruction-tuned models work better
- **Practical experience** with on-device fine-tuning

## 🔬 Technical Details

### LoRA Configuration
- **Rank (r)**: 16 - Controls the rank of the low-rank matrices
- **Alpha**: 32 - Scaling parameter for LoRA weights
- **Dropout**: 0.1 - Regularization during training
- **Target Modules**: ["q_proj", "k_proj", "v_proj", "o_proj"] - TinyLlama attention layers

### Training Parameters
- **Learning Rate**: 2e-4 - Optimized for TinyLlama
- **Batch Size**: 1 - Memory efficient for consumer hardware
- **Max Steps**: 15 - Quick demo (3-5 minutes)
- **Model**: TinyLlama-1.1B-Chat - Instruction-tuned base model

## 🚀 Next Steps

Once you've completed this demo, you can:

- **Experiment with different LoRA configurations**
- **Try QLoRA for even more memory efficiency**
- **Fine-tune on domain-specific datasets**
- **Integrate with your Hero Project**
- **Explore other PEFT methods** (AdaLoRA, IA³, etc.)

---

*This demo is part of "Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence."*