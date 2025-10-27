#!/bin/bash

# Chapter 13: RAG On-Device Setup Script
# On-Device AI: The Small Language Models Revolution

set -e  # Exit on any error

echo "🚀 Setting up Chapter 13: RAG On-Device"
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION+ is required, but you have $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "🔧 Installing Jupyter kernel..."
python -m ipykernel install --user --name=venv --display-name="Python (venv)"

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/sample_documents
mkdir -p data/embeddings
mkdir -p data/vector_stores

# Create sample documents
echo "📄 Creating sample documents..."
cat > data/sample_documents/ai_basics.txt << 'EOF'
Artificial Intelligence (AI) is a branch of computer science that aims to create 
intelligent machines that can perform tasks that typically require human intelligence. 
These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that focuses on algorithms that can learn from data 
without being explicitly programmed. Deep Learning is a subset of machine learning that 
uses neural networks with multiple layers to model and understand complex patterns.

Key areas of AI include:
- Natural Language Processing (NLP)
- Computer Vision
- Robotics
- Expert Systems
- Neural Networks
EOF

cat > data/sample_documents/small_language_models.txt << 'EOF'
Small Language Models (SLMs) are compact versions of large language models designed 
to run efficiently on local devices. They typically have fewer than 10 billion parameters 
and are optimized for speed and memory efficiency.

Key advantages of SLMs include:
- Local processing and privacy
- Lower computational requirements
- Faster inference times
- Reduced memory usage
- Cost-effective deployment

Popular SLMs include TinyLlama, Phi-3, Qwen2.5, and Gemma-2B.

SLMs are particularly useful for:
- On-device AI applications
- Edge computing scenarios
- Privacy-sensitive applications
- Resource-constrained environments
EOF

cat > data/sample_documents/rag_systems.txt << 'EOF'
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
with text generation. RAG systems retrieve relevant documents from a knowledge base and 
use them as context for generating more accurate and informed responses.

RAG systems typically consist of:
1. A document store or knowledge base
2. A retrieval system (vector search)
3. A generation model (language model)
4. A ranking and filtering mechanism

Benefits of RAG include improved accuracy, reduced hallucinations, and the ability to 
incorporate up-to-date information without retraining the model.

RAG is particularly effective for:
- Question answering systems
- Document summarization
- Knowledge-intensive tasks
- Domain-specific applications
EOF

echo "✅ Sample documents created"

# Test installation
echo "🧪 Testing installation..."

# Test imports
python3 -c "
import chromadb
import faiss
import sentence_transformers
import numpy as np
import pandas as pd
print('✅ All core dependencies imported successfully')
"

# Test ChromaDB
python3 -c "
import chromadb
client = chromadb.Client()
collection = client.create_collection('test')
collection.add(documents=['test document'], ids=['1'])
results = collection.query(query_texts=['test'], n_results=1)
print('✅ ChromaDB test successful')
"

# Test Faiss
python3 -c "
import faiss
import numpy as np
d = 128  # dimension
index = faiss.IndexFlatL2(d)
vectors = np.random.random((100, d)).astype('float32')
index.add(vectors)
distances, indices = index.search(np.random.random((1, d)).astype('float32'), 5)
print('✅ Faiss test successful')
"

# Test sentence-transformers
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['test sentence'])
print('✅ Sentence-transformers test successful')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Launch Jupyter: jupyter notebook rag_on_device_demo.ipynb"
echo "2. Select 'Python (venv)' kernel in Jupyter"
echo "3. Follow the notebook to explore RAG on-device"
echo ""
echo "🔧 Troubleshooting:"
echo "- If you see import errors, make sure you're in the venv: source venv/bin/activate"
echo "- If Jupyter doesn't show the venv kernel, run: python -m ipykernel install --user --name=venv --display-name='Python (venv)'"
echo ""
echo "📚 Documentation: README.md"
echo "🎯 Demo: rag_on_device_demo.ipynb"
