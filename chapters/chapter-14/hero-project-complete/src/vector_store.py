"""
Vector Store using ChromaDB
Hero Project: On-Device AI Agent with Vision and RAG
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import json
from typing import List, Dict, Any, Optional

class VectorStore:
    """
    ChromaDB-based vector store for RAG applications
    """
    
    def __init__(self, persist_directory="./chroma_db", collection_name="knowledge_base"):
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
        
        print(f"✅ Vector store initialized: {collection_name}")
        print(f"   Persist directory: {persist_directory}")
        print(f"   Embedding model: all-MiniLM-L6-v2")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(documents)} documents to vector store")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise
    
    def search(self, query: str, n_results: int = 3, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Search results dictionary
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"] if include_metadata else ["documents", "distances"]
            )
            
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] and include_metadata else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "query": query,
                "count": len(results["documents"][0]) if results["documents"] else 0
            }
            
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": [], "query": query, "count": 0}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self):
        """
        Clear all documents from the collection
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print("✅ Collection cleared")
        except Exception as e:
            print(f"❌ Error clearing collection: {str(e)}")
    
    def load_documents_from_directory(self, directory_path: str, file_extensions: List[str] = [".txt", ".md"]):
        """
        Load documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to include
        """
        documents = []
        metadatas = []
        ids = []
        
        try:
            for filename in os.listdir(directory_path):
                if any(filename.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(directory_path, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(content)
                    metadatas.append({
                        "filename": filename,
                        "file_path": file_path,
                        "file_type": os.path.splitext(filename)[1]
                    })
                    ids.append(f"file_{filename}")
            
            if documents:
                self.add_documents(documents, metadatas, ids)
                print(f"✅ Loaded {len(documents)} documents from {directory_path}")
            else:
                print(f"⚠️ No documents found in {directory_path}")
                
        except Exception as e:
            print(f"❌ Error loading documents from directory: {str(e)}")
    
    def save_sample_documents(self, data_directory: str = "./data/knowledge_base"):
        """
        Create sample documents for testing
        """
        os.makedirs(data_directory, exist_ok=True)
        
        sample_docs = {
            "ai_basics.txt": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that can perform tasks that typically require human intelligence. 
            These tasks include learning, reasoning, problem-solving, perception, and language understanding.
            
            Machine Learning is a subset of AI that focuses on algorithms that can learn from data 
            without being explicitly programmed. Deep Learning is a subset of machine learning that 
            uses neural networks with multiple layers to model and understand complex patterns.
            """,
            
            "small_language_models.txt": """
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
            """,
            
            "rag_systems.txt": """
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
            """,
            
            "vision_language_models.txt": """
            Vision-Language Models (VLMs) are AI models that can process and understand both 
            visual and textual information. They can analyze images, answer questions about visual 
            content, and generate descriptions or captions.
            
            Applications of VLMs include:
            - Image captioning and description
            - Visual question answering
            - Document analysis and OCR
            - Medical image analysis
            - Autonomous vehicle perception
            
            Popular VLMs include GPT-4V, Qwen2-VL, LLaVA, and CLIP.
            """
        }
        
        for filename, content in sample_docs.items():
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        print(f"✅ Created {len(sample_docs)} sample documents in {data_directory}")
        
        # Load the sample documents
        self.load_documents_from_directory(data_directory)


