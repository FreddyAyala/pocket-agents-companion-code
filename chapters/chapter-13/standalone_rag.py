"""
Standalone RAG Implementation
Chapter 13: RAG On-Device
Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence

This module provides a comprehensive RAG implementation that can work with
multiple vector databases (ChromaDB, Faiss, SQLite) for different use cases.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
import logging

# Vector database imports
import chromadb
from chromadb.utils import embedding_functions
import faiss
from sentence_transformers import SentenceTransformer

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document chunking and preprocessing for RAG systems
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def chunk_text(self, text: str, strategy: str = "fixed") -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces for embedding
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy ("fixed", "semantic", "sentence")
            
        Returns:
            List of chunk dictionaries with text, metadata, and position
        """
        chunks = []
        
        if strategy == "fixed":
            chunks = self._fixed_size_chunking(text)
        elif strategy == "semantic":
            chunks = self._semantic_chunking(text)
        elif strategy == "sentence":
            chunks = self._sentence_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        return chunks
    
    def _fixed_size_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Fixed-size chunking with overlap"""
        words = word_tokenize(text)
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "start_pos": i,
                "end_pos": min(i + self.chunk_size, len(words)),
                "chunk_id": f"chunk_{i}",
                "strategy": "fixed"
            })
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Semantic chunking based on sentence boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(word_tokenize(sentence))
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_pos": len(chunks),
                    "end_pos": len(chunks) + 1,
                    "chunk_id": f"semantic_{len(chunks)}",
                    "strategy": "semantic"
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(word_tokenize(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_pos": len(chunks),
                "end_pos": len(chunks) + 1,
                "chunk_id": f"semantic_{len(chunks)}",
                "strategy": "semantic"
            })
        
        return chunks
    
    def _sentence_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Sentence-based chunking"""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), self.chunk_size):
            chunk_sentences = sentences[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_sentences)
            
            chunks.append({
                "text": chunk_text,
                "start_pos": i,
                "end_pos": min(i + self.chunk_size, len(sentences)),
                "chunk_id": f"sentence_{i}",
                "strategy": "sentence"
            })
        
        return chunks


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using various models
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized embedding model: {model_name} (dimension: {self.dimension})")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode([text])[0]


class ChromaDBStore:
    """
    ChromaDB-based vector store for RAG applications
    """
    
    def __init__(self, persist_directory: str = "./data/vector_stores/chromadb", 
                 collection_name: str = "documents"):
        """
        Initialize ChromaDB store
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use local sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"ChromaDB store initialized: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, 
                     ids: List[str] = None):
        """Add documents to the store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "query": query,
            "count": len(results["documents"][0]) if results["documents"] else 0
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get store information"""
        count = self.collection.count()
        return {
            "type": "ChromaDB",
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }


class FaissStore:
    """
    Faiss-based vector store for high-performance search
    """
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Initialize Faiss store
        
        Args:
            dimension: Embedding dimension
            index_type: Type of Faiss index ("flat", "ivf")
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Store metadata
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        logger.info(f"Faiss store initialized: {index_type} (dimension: {dimension})")
    
    def add_embeddings(self, embeddings: np.ndarray, documents: List[str], 
                      metadatas: List[Dict] = None, ids: List[str] = None):
        """Add embeddings and documents to the store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas or [{}] * len(documents))
        self.ids.extend(ids)
        
        logger.info(f"Added {len(documents)} documents to Faiss")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), n_results)
        
        results = {
            "documents": [self.documents[i] for i in indices[0] if i < len(self.documents)],
            "metadatas": [self.metadatas[i] for i in indices[0] if i < len(self.metadatas)],
            "distances": distances[0].tolist(),
            "indices": indices[0].tolist(),
            "count": len(indices[0])
        }
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get store information"""
        return {
            "type": "Faiss",
            "index_type": self.index_type,
            "dimension": self.dimension,
            "document_count": len(self.documents),
            "index_size": self.index.ntotal
        }


class SQLiteStore:
    """
    SQLite-based vector store for minimal footprint
    """
    
    def __init__(self, db_path: str = "./data/vector_stores/sqlite.db"):
        """
        Initialize SQLite store
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                embedding BLOB
            )
        ''')
        
        self.conn.commit()
        logger.info(f"SQLite store initialized: {db_path}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, 
                     metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents and embeddings to the store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        for i, (doc, emb, meta, doc_id) in enumerate(zip(
            documents, embeddings, metadatas or [{}] * len(documents), ids
        )):
            self.cursor.execute('''
                INSERT OR REPLACE INTO documents (id, text, metadata, embedding)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, doc, json.dumps(meta), emb.tobytes()))
        
        self.conn.commit()
        logger.info(f"Added {len(documents)} documents to SQLite")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents using cosine similarity"""
        # Get all embeddings
        self.cursor.execute('SELECT id, text, metadata, embedding FROM documents')
        rows = self.cursor.fetchall()
        
        if not rows:
            return {"documents": [], "metadatas": [], "distances": [], "count": 0}
        
        # Calculate similarities
        similarities = []
        for row in rows:
            doc_id, text, metadata, embedding_bytes = row
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            similarities.append((similarity, doc_id, text, metadata))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:n_results]
        
        return {
            "documents": [result[2] for result in top_results],
            "metadatas": [json.loads(result[3]) for result in top_results],
            "distances": [1 - result[0] for result in top_results],  # Convert to distance
            "count": len(top_results)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get store information"""
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        count = self.cursor.fetchone()[0]
        
        return {
            "type": "SQLite",
            "document_count": count,
            "db_path": self.db_path
        }


class RAGSystem:
    """
    Complete RAG system that combines document processing, embedding generation,
    and vector storage for on-device question answering
    """
    
    def __init__(self, vector_store_type: str = "chromadb", **kwargs):
        """
        Initialize RAG system
        
        Args:
            vector_store_type: Type of vector store ("chromadb", "faiss", "sqlite")
            **kwargs: Additional arguments for vector store initialization
        """
        self.vector_store_type = vector_store_type
        
        # Initialize components
        self.processor = DocumentProcessor()
        self.embedder = EmbeddingGenerator()
        
        # Initialize vector store
        if vector_store_type == "chromadb":
            self.vector_store = ChromaDBStore(**kwargs)
        elif vector_store_type == "faiss":
            self.vector_store = FaissStore(dimension=self.embedder.dimension, **kwargs)
        elif vector_store_type == "sqlite":
            self.vector_store = SQLiteStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {vector_store_type}")
        
        logger.info(f"RAG system initialized with {vector_store_type}")
    
    def add_documents(self, documents: List[str], chunking_strategy: str = "semantic"):
        """
        Add documents to the RAG system
        
        Args:
            documents: List of documents to add
            chunking_strategy: Strategy for chunking documents
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, document in enumerate(documents):
            # Chunk the document
            chunks = self.processor.chunk_text(document, strategy=chunking_strategy)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk["text"])
                all_metadatas.append({
                    "document_id": f"doc_{doc_idx}",
                    "chunk_id": chunk["chunk_id"],
                    "strategy": chunk["strategy"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"]
                })
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedder.generate_embeddings(all_chunks)
        
        # Add to vector store
        if self.vector_store_type == "chromadb":
            self.vector_store.add_documents(all_chunks, all_metadatas, all_ids)
        else:
            self.vector_store.add_embeddings(embeddings, all_chunks, all_metadatas, all_ids)
        
        logger.info(f"Added {len(all_chunks)} chunks to {self.vector_store_type}")
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results with documents, metadata, and scores
        """
        if self.vector_store_type == "chromadb":
            return self.vector_store.search(query, n_results)
        else:
            # Generate query embedding
            query_embedding = self.embedder.get_embedding(query)
            return self.vector_store.search(query_embedding, n_results)
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "vector_store": self.vector_store.get_info(),
            "embedder": {
                "model": self.embedder.model_name,
                "dimension": self.embedder.dimension
            },
            "processor": {
                "chunk_size": self.processor.chunk_size,
                "overlap": self.processor.overlap
            }
        }


def demo_rag_systems():
    """
    Demonstrate different RAG system implementations
    """
    print("🚀 RAG On-Device Demo")
    print("===================")
    
    # Sample documents
    documents = [
        "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
        "Machine Learning is a subset of AI that focuses on algorithms that can learn from data.",
        "Deep Learning uses neural networks with multiple layers to model complex patterns.",
        "Natural Language Processing (NLP) enables computers to understand and process human language.",
        "Computer Vision allows machines to interpret and understand visual information from images."
    ]
    
    # Test different vector stores
    stores = ["chromadb", "faiss", "sqlite"]
    
    for store_type in stores:
        print(f"\n📊 Testing {store_type.upper()} RAG System")
        print("-" * 40)
        
        try:
            # Initialize RAG system
            rag = RAGSystem(vector_store_type=store_type)
            
            # Add documents
            rag.add_documents(documents, chunking_strategy="semantic")
            
            # Test search
            query = "What is machine learning?"
            results = rag.search(query, n_results=3)
            
            print(f"Query: {query}")
            print(f"Results: {results['count']} documents found")
            for i, doc in enumerate(results['documents']):
                print(f"  {i+1}. {doc[:100]}...")
            
            # System info
            info = rag.get_info()
            print(f"System info: {info}")
            
        except Exception as e:
            print(f"❌ Error with {store_type}: {str(e)}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    demo_rag_systems()
