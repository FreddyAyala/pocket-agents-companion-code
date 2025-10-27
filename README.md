# Pocket Agents: Companion Code

**Companion code repository for "Pocket Agents: A Practical Guide to On-Device Artificial Intelligence"**

This repository contains all the practical code examples, demonstrations, and implementations from the book, organized by chapter for easy navigation and learning.

## 📚 Book Overview

"Pocket Agents" is a comprehensive guide to building autonomous, privacy-first AI agents that run entirely on your device. This companion code provides hands-on implementations of every concept covered in the book.

## 🚀 Quick Start

### System Requirements
- Python 3.8+ (Python 3.11+ recommended)
- 8GB+ RAM (16GB+ for Hero Project)
- macOS, Linux, or Windows
- Git

### General Setup
```bash
# Clone the repository
git clone <repository-url>
cd pocket-agents-companion-code

# Each chapter has its own setup - see individual chapter READMEs
```

## 📋 Chapter Guide

| Chapter | Title | Key Files | Dependencies | Setup Time |
|---------|-------|-----------|--------------|------------|
| **04** | Performance Metrics | `metrics_demo.ipynb` | numpy, matplotlib | 5 min |
| **05** | Model Compression | `quantization_demo.ipynb` | torch, transformers | 10 min |
| **06** | Pruning & Distillation | `pruning_sparsity_demo.ipynb` | torch, transformers | 10 min |
| **07** | Fine-Tuning & Adaptation | `lora_finetuning_demo.ipynb` | torch, peft, datasets | 15 min |
| **08** | Inference Engines | `inference_engines_demo.ipynb` | llama-cpp-python, onnx | 10 min |
| **09** | Hardware Battlefield | `hardware_battlefield_demo.ipynb` | psutil, GPUtil | 5 min |
| **10** | Deployment Playbook | `deployment_playbook_demo.ipynb` | fastapi, gradio | 10 min |
| **11** | Edge Management | `edge_management_demo.ipynb` | sqlite3, threading | 5 min |
| **12** | Agentic Patterns | `agentic_workflows.ipynb` | jupyter, basic Python | 5 min |
| **13** | RAG On-Device | `rag_on_device_demo.ipynb` | chromadb, sentence-transformers | 15 min |
| **14** | Hero Project | `hero-project-complete/` | torch, transformers, llama-cpp-python | 30 min |

## 🎯 Learning Path

### For Beginners
1. Start with **Chapter 4** (Performance Metrics) to understand the fundamentals
2. Progress through **Chapters 5-8** to learn model optimization
3. Explore **Chapters 9-11** for deployment and edge management
4. Master **Chapter 12** (Agentic Patterns) for AI agent concepts
5. Build the **Hero Project** in **Chapter 14** for a complete system

### For Experienced Developers
- Jump directly to **Chapter 12** for agentic patterns
- Implement the **Hero Project** from **Chapter 14**
- Use **Chapters 5-8** for model optimization techniques
- Reference **Chapters 9-11** for production deployment

## 🛠️ Chapter Structure

Each chapter follows a consistent structure:

```
chapter-XX/
├── README.md              # Chapter overview and setup
├── requirements.txt        # Python dependencies
├── setup_and_test.sh      # Automated setup script
├── demo.ipynb            # Interactive Jupyter notebook
├── example.py            # Standalone Python examples
└── data/                 # Sample data (where applicable)
```

## 🚀 Special Implementations

### Agentic Patterns Demo (Chapter 12)
Educational implementation with mock responses - perfect for learning concepts:
```bash
cd agentic-patterns-demo/
jupyter notebook agentic_workflows.ipynb
```

### Hero Project (Chapter 14)
Complete production-ready AI agent with real model loading:
```bash
cd chapters/chapter-14/hero-project-complete/
./setup_and_test.sh
./launch_hero_ui.sh
```

## 📖 Book Integration

This companion code is designed to work alongside the book:

- **Theory** → Read the book chapters
- **Practice** → Run the companion code
- **Build** → Implement your own projects
- **Deploy** → Use the Hero Project as a foundation

## 🔧 Troubleshooting

### Common Issues

**Import Errors**: Make sure you're in the correct virtual environment
```bash
source venv/bin/activate  # or your preferred environment
```

**Memory Issues**: The Hero Project requires significant RAM
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM
- For smaller systems: Use the agentic patterns demo instead

**Path Issues**: Some notebooks may need path adjustments
- Check the notebook's first cell for setup instructions
- Adjust paths based on your system

### Getting Help

1. Check the chapter-specific README
2. Review the setup script output
3. Ensure all dependencies are installed
4. Verify Python version compatibility

## 📄 License

This companion code is released under the same license as the book. See individual chapter files for specific licensing information.

## 🤝 Contributing

This repository is designed to accompany the book. For issues or improvements:

1. Check if the issue is covered in the book
2. Verify you're using the correct Python version
3. Ensure all dependencies are properly installed
4. Create an issue with detailed information

## 📚 Book Information

**Title**: Pocket Agents: A Practical Guide to On-Device Artificial Intelligence  
**Author**: [Author Name]  
**Publisher**: [Publisher Name]  
**ISBN**: [ISBN Number]  

For the complete book content, purchase from your preferred retailer.

---

*This companion code repository provides hands-on implementations of every concept in "Pocket Agents". Start with the fundamentals and build your way up to creating your own on-device AI agents.*