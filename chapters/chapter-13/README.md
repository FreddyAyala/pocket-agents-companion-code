# Chapter 13: RAG On-Device

**Pocket Agents: A Practical Guide to On-Device Artificial Intelligence**

This companion code demonstrates Retrieval-Augmented Generation (RAG) systems that run entirely on-device, combining local vector databases with small language models for private, intelligent question answering.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to this directory
cd companion-code/chapters/chapter-13

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
jupyter notebook rag_on_device_demo.ipynb
```

## 📋 What You'll Learn

- **Vector Database Comparison**: ChromaDB, Faiss, and SQLite for different use cases
- **Embedding Generation**: Local embedding models and optimization strategies
- **Document Chunking**: Fixed-size vs semantic chunking approaches
- **Hybrid Search**: Combining vector and keyword search for better results
- **RAG Pipeline**: Complete on-device RAG system implementation
- **Performance Optimization**: Memory, speed, and accuracy trade-offs

## 🎯 Key Concepts

- **Vector Databases**: ChromaDB (simplicity), Faiss (speed), SQLite (minimal footprint)
- **Embeddings**: all-MiniLM-L6-v2 for local processing
- **Chunking Strategies**: Fixed-size, semantic, and overlap techniques
- **Search Optimization**: Hybrid ranking and result deduplication
- **Local Processing**: Complete privacy with no cloud dependencies

## 🔬 Techniques Demonstrated

### 1. Vector Database Comparison
- **ChromaDB**: Easy setup, good for prototyping
- **Faiss**: High-performance indexing for production
- **SQLite**: Minimal footprint for embedded systems

### 2. Embedding Generation
- Local sentence-transformers models
- Memory-efficient batch processing
- Embedding quality comparison

### 3. Document Processing
- Text chunking strategies
- Metadata extraction
- Content preprocessing

### 4. Search and Retrieval
- Vector similarity search
- Keyword search integration
- Result ranking and filtering

### 5. RAG Pipeline
- Context retrieval
- Prompt construction
- Response generation
- Source attribution

## 📊 Expected Results

- **ChromaDB**: Easy setup, good for small-medium datasets
- **Faiss**: 10-100x faster search for large datasets
- **SQLite**: Minimal memory footprint (< 50MB)
- **Embeddings**: 384-dimensional vectors for semantic search
- **RAG Quality**: Context-aware responses with source attribution

## 🚀 Performance Benefits

- **Complete Privacy**: No data leaves your device
- **Fast Retrieval**: Sub-second search across thousands of documents
- **Low Memory**: Efficient storage and processing
- **Scalable**: Works from hundreds to millions of documents
- **Offline Capable**: No internet required for operation

## ⚖️ Trade-offs

- **Memory Usage**: Vector storage requires RAM/disk space
- **Processing Time**: Embedding generation can be slow for large documents
- **Quality vs Speed**: More sophisticated chunking takes longer
- **Model Size**: Larger embedding models provide better quality

## 🔗 Related Chapters

- Chapter 5: Model Compression - Quantization
- Chapter 7: Fine-Tuning & Adaptation
- Chapter 12: The Hero Project (Capstone)
- Chapter 14: Agentic Best Practices

## 💡 Best Practices

1. **Start Simple**: Begin with ChromaDB for prototyping
2. **Choose Right Embedding**: Balance quality vs speed for your use case
3. **Optimize Chunking**: Experiment with different chunk sizes and overlap
4. **Monitor Performance**: Track memory usage and search speed
5. **Test Quality**: Validate RAG responses against ground truth

## 🛠️ Files in this Chapter

- `rag_on_device_demo.ipynb` - Comprehensive RAG demonstration
- `standalone_rag.py` - Production-ready RAG implementation
- `embedding_comparison.py` - Embedding model comparison
- `chunking_strategies.py` - Document chunking techniques
- `hybrid_search.py` - Advanced search strategies

## 🎮 Interactive Demo

The Jupyter notebook provides a step-by-step walkthrough:

1. **Vector Database Setup**: Initialize ChromaDB, Faiss, and SQLite
2. **Document Processing**: Load and chunk sample documents
3. **Embedding Generation**: Create vector representations
4. **Search Testing**: Compare different search strategies
5. **RAG Pipeline**: Complete question-answering system
6. **Performance Analysis**: Benchmark different approaches

## 🔧 Troubleshooting

### Common Issues

- **"ChromaDB not found"**: Run `pip install chromadb`
- **"Out of memory"**: Reduce chunk size or use smaller embedding model
- **"Slow search"**: Try Faiss for better performance
- **"Poor results"**: Adjust chunk size or try different embedding model

### Performance Tips

1. **Use Faiss for large datasets** (>10k documents)
2. **Optimize chunk size** (512-1024 tokens typically work well)
3. **Batch embedding generation** for better efficiency
4. **Use quantized models** for memory-constrained environments

---

*This chapter demonstrates how to build production-ready RAG systems that run entirely on-device, ensuring complete privacy while maintaining high performance.*
