"""
Document Chunking Strategies
Chapter 13: RAG On-Device
Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence

This module demonstrates different document chunking strategies for RAG applications,
including fixed-size, semantic, and sentence-based chunking approaches.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


class ChunkingAnalyzer:
    """
    Analyze and compare different chunking strategies
    """
    
    def __init__(self):
        """Initialize chunking analyzer"""
        self.stop_words = set(stopwords.words('english'))
    
    def fixed_size_chunking(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[Dict[str, Any]]:
        """
        Fixed-size chunking with overlap
        
        Args:
            text: Input text to chunk
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        words = word_tokenize(text)
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "start_pos": i,
                "end_pos": min(i + chunk_size, len(words)),
                "chunk_id": f"fixed_{i}",
                "strategy": "fixed",
                "word_count": len(chunk_words),
                "char_count": len(chunk_text)
            })
        
        return chunks
    
    def sentence_chunking(self, text: str, sentences_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Sentence-based chunking
        
        Args:
            text: Input text to chunk
            sentences_per_chunk: Number of sentences per chunk
            
        Returns:
            List of chunk dictionaries
        """
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)
            
            chunks.append({
                "text": chunk_text,
                "start_pos": i,
                "end_pos": min(i + sentences_per_chunk, len(sentences)),
                "chunk_id": f"sentence_{i}",
                "strategy": "sentence",
                "sentence_count": len(chunk_sentences),
                "word_count": len(word_tokenize(chunk_text)),
                "char_count": len(chunk_text)
            })
        
        return chunks
    
    def semantic_chunking(self, text: str, max_chunk_size: int = 200, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Semantic chunking based on content similarity
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum words per chunk
            similarity_threshold: Threshold for combining chunks
            
        Returns:
            List of chunk dictionaries
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence)
            
            # If adding this sentence would exceed max size, create a chunk
            if current_size + len(sentence_words) > max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_pos": len(chunks),
                    "end_pos": len(chunks) + 1,
                    "chunk_id": f"semantic_{len(chunks)}",
                    "strategy": "semantic",
                    "sentence_count": len(current_chunk),
                    "word_count": len(word_tokenize(chunk_text)),
                    "char_count": len(chunk_text)
                })
                current_chunk = [sentence]
                current_size = len(sentence_words)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence_words)
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_pos": len(chunks),
                "end_pos": len(chunks) + 1,
                "chunk_id": f"semantic_{len(chunks)}",
                "strategy": "semantic",
                "sentence_count": len(current_chunk),
                "word_count": len(word_tokenize(chunk_text)),
                "char_count": len(chunk_text)
            })
        
        return chunks
    
    def paragraph_chunking(self, text: str) -> List[Dict[str, Any]]:
        """
        Paragraph-based chunking
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries
        """
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():  # Skip empty paragraphs
                chunks.append({
                    "text": paragraph.strip(),
                    "start_pos": i,
                    "end_pos": i + 1,
                    "chunk_id": f"paragraph_{i}",
                    "strategy": "paragraph",
                    "word_count": len(word_tokenize(paragraph)),
                    "char_count": len(paragraph)
                })
        
        return chunks
    
    def analyze_chunk_quality(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Quality analysis results
        """
        if not chunks:
            return {}
        
        # Extract metrics
        word_counts = [chunk.get('word_count', 0) for chunk in chunks]
        char_counts = [chunk.get('char_count', 0) for chunk in chunks]
        
        # Calculate statistics
        analysis = {
            "total_chunks": len(chunks),
            "avg_words_per_chunk": np.mean(word_counts),
            "std_words_per_chunk": np.std(word_counts),
            "min_words": np.min(word_counts),
            "max_words": np.max(word_counts),
            "avg_chars_per_chunk": np.mean(char_counts),
            "std_chars_per_chunk": np.std(char_counts),
            "chunk_size_variance": np.var(word_counts),
            "chunk_size_coefficient": np.std(word_counts) / np.mean(word_counts) if np.mean(word_counts) > 0 else 0
        }
        
        return analysis
    
    def compare_chunking_strategies(self, text: str) -> Dict[str, Any]:
        """
        Compare different chunking strategies on the same text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Comparison results
        """
        strategies = {
            "fixed_100": self.fixed_size_chunking(text, chunk_size=100, overlap=20),
            "fixed_200": self.fixed_size_chunking(text, chunk_size=200, overlap=40),
            "sentence_2": self.sentence_chunking(text, sentences_per_chunk=2),
            "sentence_3": self.sentence_chunking(text, sentences_per_chunk=3),
            "semantic_200": self.semantic_chunking(text, max_chunk_size=200),
            "paragraph": self.paragraph_chunking(text)
        }
        
        comparison = {}
        
        for strategy_name, chunks in strategies.items():
            analysis = self.analyze_chunk_quality(chunks)
            comparison[strategy_name] = {
                "chunks": chunks,
                "analysis": analysis
            }
        
        return comparison
    
    def visualize_chunking_comparison(self, comparison: Dict[str, Any], save_path: str = None):
        """
        Create visualization comparing chunking strategies
        
        Args:
            comparison: Comparison results
            save_path: Path to save plot (optional)
        """
        strategies = list(comparison.keys())
        metrics = ['total_chunks', 'avg_words_per_chunk', 'chunk_size_variance']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Chunking Strategy Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            values = [comparison[strategy]['analysis'].get(metric, 0) for strategy in strategies]
            axes[i].bar(strategies, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def create_chunking_recommendations(self, comparison: Dict[str, Any]) -> str:
        """
        Create recommendations based on chunking analysis
        
        Args:
            comparison: Comparison results
            
        Returns:
            Recommendation report
        """
        report = []
        report.append("🎯 Chunking Strategy Recommendations")
        report.append("=" * 40)
        
        # Find best strategies for different criteria
        best_consistency = min(comparison.keys(), 
                             key=lambda x: comparison[x]['analysis'].get('chunk_size_variance', float('inf')))
        most_chunks = max(comparison.keys(), 
                         key=lambda x: comparison[x]['analysis'].get('total_chunks', 0))
        most_balanced = min(comparison.keys(), 
                           key=lambda x: comparison[x]['analysis'].get('chunk_size_coefficient', float('inf')))
        
        report.append(f"\n🏆 Best Strategies:")
        report.append(f"  Most Consistent: {best_consistency}")
        report.append(f"  Most Granular: {most_chunks}")
        report.append(f"  Most Balanced: {most_balanced}")
        
        # Use case recommendations
        report.append(f"\n💡 Use Case Recommendations:")
        report.append(f"  📄 Long Documents: semantic_200 (preserves context)")
        report.append(f"  🔍 Search Optimization: fixed_100 (consistent chunks)")
        report.append(f"  📱 Mobile/Edge: fixed_200 (fewer chunks)")
        report.append(f"  📚 Structured Content: paragraph (natural boundaries)")
        
        # General guidelines
        report.append(f"\n📋 General Guidelines:")
        report.append(f"  • Chunk size: 100-500 words typically work well")
        report.append(f"  • Overlap: 10-20% for fixed-size chunking")
        report.append(f"  • Consider content type: technical vs narrative")
        report.append(f"  • Test with your specific use case")
        
        return "\n".join(report)


def demo_chunking_strategies():
    """
    Demonstrate different chunking strategies
    """
    print("📄 Chunking Strategies Demo")
    print("=" * 30)
    
    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.
    
    Machine Learning is a subset of AI that focuses on algorithms that can learn from data 
    without being explicitly programmed. Deep Learning is a subset of machine learning that 
    uses neural networks with multiple layers to model and understand complex patterns.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
    between computers and humans through natural language. The ultimate objective of NLP 
    is to read, decipher, understand, and make sense of human language in a valuable way.
    
    Computer Vision is a field of AI that trains computers to interpret and understand 
    the visual world. Using digital images from cameras and videos and deep learning models, 
    machines can accurately identify and classify objects and then react to what they see.
    
    Robotics is an interdisciplinary field that integrates computer science and engineering. 
    Robotics involves the design, construction, operation, and use of robots. The goal of 
    robotics is to design machines that can help and assist humans.
    """
    
    # Initialize analyzer
    analyzer = ChunkingAnalyzer()
    
    print(f"Analyzing text with {len(sample_text)} characters")
    
    # Compare strategies
    comparison = analyzer.compare_chunking_strategies(sample_text)
    
    # Display results
    print("\n📊 Chunking Results:")
    for strategy, data in comparison.items():
        analysis = data['analysis']
        print(f"\n{strategy}:")
        print(f"  Chunks: {analysis['total_chunks']}")
        print(f"  Avg words: {analysis['avg_words_per_chunk']:.1f}")
        print(f"  Variance: {analysis['chunk_size_variance']:.1f}")
    
    # Create visualization
    analyzer.visualize_chunking_comparison(comparison)
    
    # Generate recommendations
    recommendations = analyzer.create_chunking_recommendations(comparison)
    print(f"\n{recommendations}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    demo_chunking_strategies()
