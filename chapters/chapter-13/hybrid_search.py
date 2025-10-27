"""
Hybrid Search Implementation
Chapter 13: RAG On-Device
Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence

This module implements hybrid search combining vector similarity and keyword matching
for improved retrieval in RAG systems.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class HybridSearchEngine:
    """
    Hybrid search engine combining vector similarity and keyword matching
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid search engine
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.stop_words = set(stopwords.words('english'))
        
        # Storage for documents and embeddings
        self.documents = []
        self.document_embeddings = None
        self.tfidf_matrix = None
        self.document_metadata = []
        
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Add documents to the search index
        
        Args:
            documents: List of documents to index
            metadatas: Optional metadata for each document
        """
        self.documents.extend(documents)
        
        if metadatas is None:
            metadatas = [{}] * len(documents)
        self.document_metadata.extend(metadatas)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents)
        
        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
        
        # Generate TF-IDF matrix
        print("Computing TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        
        print(f"✅ Indexed {len(self.documents)} documents")
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        query_embedding = self.embedding_model.encode([query])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform keyword-based search using TF-IDF
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_index, tfidf_score) tuples
        """
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword scores
        
        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword similarity (0-1)
            
        Returns:
            List of search results with combined scores
        """
        # Get vector search results
        vector_results = self.vector_search(query, top_k * 2)  # Get more for combination
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Create score dictionaries
        vector_scores = {idx: score for idx, score in vector_results}
        keyword_scores = {idx: score for idx, score in keyword_results}
        
        # Combine scores
        combined_scores = {}
        all_indices = set(vector_scores.keys()) | set(keyword_scores.keys())
        
        for idx in all_indices:
            vector_score = vector_scores.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            
            # Normalize scores to 0-1 range
            combined_score = (vector_weight * vector_score + 
                            keyword_weight * keyword_score)
            combined_scores[idx] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for idx, score in sorted_results[:top_k]:
            results.append({
                "document_index": idx,
                "document": self.documents[idx],
                "metadata": self.document_metadata[idx],
                "combined_score": score,
                "vector_score": vector_scores.get(idx, 0.0),
                "keyword_score": keyword_scores.get(idx, 0.0)
            })
        
        return results
    
    def advanced_hybrid_search(self, query: str, top_k: int = 10, 
                              boost_exact_matches: bool = True,
                              boost_phrase_matches: bool = True) -> List[Dict[str, Any]]:
        """
        Advanced hybrid search with query analysis and boosting
        
        Args:
            query: Search query
            top_k: Number of results to return
            boost_exact_matches: Whether to boost exact word matches
            boost_phrase_matches: Whether to boost phrase matches
            
        Returns:
            List of enhanced search results
        """
        # Basic hybrid search
        results = self.hybrid_search(query, top_k * 2)
        
        # Query analysis
        query_words = set(word_tokenize(query.lower()))
        query_phrases = self._extract_phrases(query)
        
        # Apply boosting
        for result in results:
            doc_text = result['document'].lower()
            doc_words = set(word_tokenize(doc_text))
            
            # Exact word match boosting
            if boost_exact_matches:
                exact_matches = len(query_words.intersection(doc_words))
                result['exact_match_boost'] = exact_matches / len(query_words)
                result['combined_score'] += result['exact_match_boost'] * 0.1
            
            # Phrase match boosting
            if boost_phrase_matches:
                phrase_matches = sum(1 for phrase in query_phrases if phrase in doc_text)
                result['phrase_match_boost'] = phrase_matches * 0.2
                result['combined_score'] += result['phrase_match_boost']
        
        # Re-sort by enhanced scores
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results[:top_k]
    
    def _extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """
        Extract meaningful phrases from text
        
        Args:
            text: Input text
            min_length: Minimum phrase length
            max_length: Maximum phrase length
            
        Returns:
            List of extracted phrases
        """
        words = word_tokenize(text.lower())
        phrases = []
        
        for length in range(min_length, max_length + 1):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i + length])
                # Filter out phrases with only stop words
                if not all(word in self.stop_words for word in words[i:i + length]):
                    phrases.append(phrase)
        
        return phrases
    
    def search_with_filters(self, query: str, filters: Dict[str, Any] = None, 
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search with metadata filters
        
        Args:
            query: Search query
            filters: Dictionary of metadata filters
            top_k: Number of results to return
            
        Returns:
            Filtered search results
        """
        # Perform hybrid search
        results = self.hybrid_search(query, top_k * 2)
        
        # Apply filters
        if filters:
            filtered_results = []
            for result in results:
                metadata = result['metadata']
                include = True
                
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        include = False
                        break
                
                if include:
                    filtered_results.append(result)
            
            results = filtered_results
        
        return results[:top_k]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the search index
        
        Returns:
            Dictionary with index statistics
        """
        if not self.documents:
            return {"error": "No documents indexed"}
        
        # Calculate statistics
        doc_lengths = [len(doc.split()) for doc in self.documents]
        
        stats = {
            "total_documents": len(self.documents),
            "avg_document_length": np.mean(doc_lengths),
            "min_document_length": np.min(doc_lengths),
            "max_document_length": np.max(doc_lengths),
            "embedding_dimension": self.document_embeddings.shape[1] if self.document_embeddings is not None else 0,
            "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_matrix is not None else 0
        }
        
        return stats


def demo_hybrid_search():
    """
    Demonstrate hybrid search capabilities
    """
    print("🔍 Hybrid Search Demo")
    print("=" * 25)
    
    # Sample documents
    documents = [
        "Artificial Intelligence is transforming healthcare with diagnostic tools and treatment recommendations.",
        "Machine Learning algorithms analyze medical data to predict patient outcomes and disease progression.",
        "Natural Language Processing helps doctors extract insights from clinical notes and research papers.",
        "Computer Vision systems can detect tumors in medical images with high accuracy.",
        "Robotics in surgery enables precise procedures with minimal invasiveness.",
        "Data Science in healthcare involves analyzing patient records and treatment effectiveness.",
        "AI-powered drug discovery accelerates the development of new medications.",
        "Telemedicine platforms use AI to provide remote healthcare services."
    ]
    
    metadatas = [
        {"category": "AI", "domain": "healthcare"},
        {"category": "ML", "domain": "healthcare"},
        {"category": "NLP", "domain": "healthcare"},
        {"category": "CV", "domain": "healthcare"},
        {"category": "robotics", "domain": "healthcare"},
        {"category": "data_science", "domain": "healthcare"},
        {"category": "AI", "domain": "pharmaceuticals"},
        {"category": "AI", "domain": "telemedicine"}
    ]
    
    # Initialize search engine
    search_engine = HybridSearchEngine()
    
    # Add documents
    search_engine.add_documents(documents, metadatas)
    
    # Test different search types
    queries = [
        "AI in healthcare",
        "machine learning medical",
        "robotic surgery",
        "drug discovery"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 30)
        
        # Vector search
        vector_results = search_engine.vector_search(query, top_k=3)
        print("Vector Search Results:")
        for idx, score in vector_results:
            print(f"  {score:.3f}: {documents[idx][:60]}...")
        
        # Keyword search
        keyword_results = search_engine.keyword_search(query, top_k=3)
        print("\nKeyword Search Results:")
        for idx, score in keyword_results:
            print(f"  {score:.3f}: {documents[idx][:60]}...")
        
        # Hybrid search
        hybrid_results = search_engine.hybrid_search(query, top_k=3)
        print("\nHybrid Search Results:")
        for result in hybrid_results:
            print(f"  {result['combined_score']:.3f}: {result['document'][:60]}...")
            print(f"    Vector: {result['vector_score']:.3f}, Keyword: {result['keyword_score']:.3f}")
    
    # Test advanced search
    print(f"\n🚀 Advanced Search Demo")
    print("-" * 30)
    
    advanced_query = "AI machine learning healthcare"
    advanced_results = search_engine.advanced_hybrid_search(advanced_query, top_k=3)
    
    print(f"Advanced Query: '{advanced_query}'")
    for result in advanced_results:
        print(f"  Score: {result['combined_score']:.3f}")
        print(f"    Document: {result['document'][:60]}...")
        print(f"    Exact matches: {result.get('exact_match_boost', 0):.3f}")
        print(f"    Phrase matches: {result.get('phrase_match_boost', 0):.3f}")
    
    # Test filtered search
    print(f"\n🔧 Filtered Search Demo")
    print("-" * 30)
    
    filtered_results = search_engine.search_with_filters(
        "AI healthcare", 
        filters={"category": "AI"}, 
        top_k=3
    )
    
    print("Filtered Results (category='AI'):")
    for result in filtered_results:
        print(f"  {result['combined_score']:.3f}: {result['document'][:60]}...")
        print(f"    Metadata: {result['metadata']}")
    
    # Display statistics
    stats = search_engine.get_search_statistics()
    print(f"\n📊 Search Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    demo_hybrid_search()
