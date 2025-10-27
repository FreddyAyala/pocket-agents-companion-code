"""
RAG Agent using Atomic Agents Framework
Hero Project: On-Device AI Agent with Vision and RAG
"""

from typing import Dict, Any, Optional, List
import logging

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent for question answering
    """
    
    def __init__(self, model_loader, vector_store, system_prompt: Optional[str] = None):
        """
        Initialize RAG Agent
        
        Args:
            model_loader: Qwen3VLLoader instance
            vector_store: VectorStore instance
            system_prompt: Custom system prompt (optional)
        """
        self.model_loader = model_loader
        self.vector_store = vector_store
        
        # Default system prompt
        default_system_prompt = """You are a helpful AI assistant with access to a knowledge base.
        Your role is to answer questions using the retrieved context from the knowledge base.
        
        Guidelines:
        - Use the provided context to answer questions accurately
        - If the context doesn't contain enough information, say so
        - Be concise but comprehensive in your responses
        - Always be helpful and professional"""
        
        self.system_prompt = system_prompt or default_system_prompt
        print("✅ RAG Agent initialized")
    
    def run(self, query: str, image=None, n_results: int = 3) -> Dict[str, Any]:
        """
        Run RAG query with optional image
        
        Args:
            query: User question
            image: Optional PIL image
            n_results: Number of retrieval results
            
        Returns:
            Dictionary with answer, context, and sources
        """
        try:
            # Retrieve context from vector store
            print(f"🔍 Searching knowledge base for: '{query}'")
            search_results = self.vector_store.search(query, n_results=n_results)
            
            # Extract context
            if search_results['documents'] and search_results['documents'][0]:
                context = "\n".join(search_results['documents'][0])
                sources = search_results['metadatas'][0] if search_results['metadatas'] else []
            else:
                context = "No relevant context found in knowledge base."
                sources = []
            
            # Build messages for the model
            messages = [
                {"role": "system", "content": f"{self.system_prompt}\n\nContext:\n{context}"},
                {"role": "user", "content": query}
            ]
            
            # Generate response using the model
            print("🤖 Generating response...")
            response = self.model_loader.generate_response(messages, images=[image] if image else None)
            
            return {
                "answer": response,
                "context": context,
                "sources": sources,
                "query": query
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "context": "",
                "sources": [],
                "query": query
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "type": "RAG Agent",
            "model": self.model_loader.model_path,
            "vector_store": "ChromaDB",
            "capabilities": ["text_qa", "image_qa", "context_retrieval"]
        }