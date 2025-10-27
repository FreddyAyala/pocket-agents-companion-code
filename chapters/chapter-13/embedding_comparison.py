"""
Embedding Model Comparison
Chapter 13: RAG On-Device
Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence

This module compares different embedding models for RAG applications,
including performance, quality, and memory usage analysis.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import psutil
import os

class EmbeddingComparison:
    """
    Compare different embedding models for RAG applications
    """
    
    def __init__(self):
        """Initialize embedding comparison"""
        self.models = {}
        self.results = {}
        
        # Popular embedding models for comparison
        self.model_configs = {
            "all-MiniLM-L6-v2": {
                "description": "Lightweight, fast model (384 dim)",
                "size": "22MB",
                "dimension": 384
            },
            "all-MiniLM-L12-v2": {
                "description": "Better quality, larger (384 dim)",
                "size": "33MB", 
                "dimension": 384
            },
            "all-mpnet-base-v2": {
                "description": "High quality, larger (768 dim)",
                "size": "420MB",
                "dimension": 768
            },
            "paraphrase-MiniLM-L6-v2": {
                "description": "Optimized for paraphrasing (384 dim)",
                "size": "22MB",
                "dimension": 384
            }
        }
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a sentence transformer model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        print(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            self.models[model_name] = {
                "model": model,
                "load_time": load_time,
                "dimension": model.get_sentence_embedding_dimension()
            }
            
            print(f"✅ {model_name} loaded in {load_time:.2f}s")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            return None
    
    def benchmark_embedding_generation(self, texts: List[str], model_name: str) -> Dict[str, Any]:
        """
        Benchmark embedding generation for a model
        
        Args:
            texts: List of texts to embed
            model_name: Name of the model to benchmark
            
        Returns:
            Benchmark results
        """
        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return None
        
        model = self.models[model_name]["model"]
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate embeddings with timing
        start_time = time.time()
        embeddings = model.encode(texts, show_progress_bar=False)
        generation_time = time.time() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Calculate statistics
        results = {
            "model_name": model_name,
            "text_count": len(texts),
            "generation_time": generation_time,
            "time_per_text": generation_time / len(texts),
            "memory_used_mb": memory_used,
            "embedding_dimension": embeddings.shape[1],
            "total_embeddings_size_mb": embeddings.nbytes / 1024 / 1024,
            "throughput_texts_per_second": len(texts) / generation_time
        }
        
        return results
    
    def compare_embedding_quality(self, texts: List[str], model_names: List[str]) -> Dict[str, Any]:
        """
        Compare embedding quality using similarity analysis
        
        Args:
            texts: List of texts to compare
            model_names: List of model names to compare
            
        Returns:
            Quality comparison results
        """
        results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]["model"]
            embeddings = model.encode(texts)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)
            
            results[model_name] = {
                "mean_similarity": np.mean(similarities),
                "std_similarity": np.std(similarities),
                "min_similarity": np.min(similarities),
                "max_similarity": np.max(similarities),
                "embedding_dimension": embeddings.shape[1]
            }
        
        return results
    
    def benchmark_all_models(self, texts: List[str], model_names: List[str] = None) -> pd.DataFrame:
        """
        Benchmark all specified models
        
        Args:
            texts: List of texts to benchmark
            model_names: List of model names (default: all configured models)
            
        Returns:
            DataFrame with benchmark results
        """
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        benchmark_results = []
        
        for model_name in model_names:
            print(f"\n🔍 Benchmarking {model_name}")
            print("-" * 50)
            
            # Load model if not already loaded
            if model_name not in self.models:
                self.load_model(model_name)
            
            if model_name in self.models:
                # Benchmark embedding generation
                results = self.benchmark_embedding_generation(texts, model_name)
                if results:
                    benchmark_results.append(results)
        
        return pd.DataFrame(benchmark_results)
    
    def plot_performance_comparison(self, df: pd.DataFrame, save_path: str = None):
        """
        Create performance comparison plots
        
        Args:
            df: DataFrame with benchmark results
            save_path: Path to save plots (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Embedding Model Performance Comparison', fontsize=16)
        
        # Throughput comparison
        axes[0, 0].bar(df['model_name'], df['throughput_texts_per_second'])
        axes[0, 0].set_title('Throughput (Texts/Second)')
        axes[0, 0].set_ylabel('Texts/Second')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        axes[0, 1].bar(df['model_name'], df['memory_used_mb'])
        axes[0, 1].set_title('Memory Usage (MB)')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Generation time per text
        axes[1, 0].bar(df['model_name'], df['time_per_text'])
        axes[1, 0].set_title('Time per Text (seconds)')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Embedding dimension
        axes[1, 1].bar(df['model_name'], df['embedding_dimension'])
        axes[1, 1].set_title('Embedding Dimension')
        axes[1, 1].set_ylabel('Dimension')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def create_recommendation_report(self, df: pd.DataFrame) -> str:
        """
        Create a recommendation report based on benchmark results
        
        Args:
            df: DataFrame with benchmark results
            
        Returns:
            Recommendation report string
        """
        report = []
        report.append("🎯 Embedding Model Recommendation Report")
        report.append("=" * 50)
        
        # Find best models for different criteria
        fastest = df.loc[df['throughput_texts_per_second'].idxmax()]
        most_memory_efficient = df.loc[df['memory_used_mb'].idxmin()]
        smallest_dimension = df.loc[df['embedding_dimension'].idxmin()]
        
        report.append(f"\n🏆 Best Performance:")
        report.append(f"  Fastest: {fastest['model_name']} ({fastest['throughput_texts_per_second']:.1f} texts/sec)")
        report.append(f"  Most Memory Efficient: {most_memory_efficient['model_name']} ({most_memory_efficient['memory_used_mb']:.1f} MB)")
        report.append(f"  Smallest Dimension: {smallest['model_name']} ({smallest['embedding_dimension']} dim)")
        
        # Recommendations for different use cases
        report.append(f"\n💡 Use Case Recommendations:")
        report.append(f"  🚀 Speed Critical: {fastest['model_name']}")
        report.append(f"  💾 Memory Constrained: {most_memory_efficient['model_name']}")
        report.append(f"  📱 Mobile/Edge: {smallest_dimension['model_name']}")
        
        # General recommendations
        report.append(f"\n📋 General Recommendations:")
        report.append(f"  • For prototyping: all-MiniLM-L6-v2 (fast, lightweight)")
        report.append(f"  • For production: all-mpnet-base-v2 (high quality)")
        report.append(f"  • For mobile: all-MiniLM-L6-v2 (smallest footprint)")
        
        return "\n".join(report)


def demo_embedding_comparison():
    """
    Demonstrate embedding model comparison
    """
    print("🔍 Embedding Model Comparison Demo")
    print("=" * 40)
    
    # Sample texts for testing
    sample_texts = [
        "Artificial Intelligence is transforming the world",
        "Machine Learning algorithms learn from data",
        "Deep Learning uses neural networks",
        "Natural Language Processing understands text",
        "Computer Vision analyzes images",
        "Robotics combines AI with physical systems",
        "Data Science extracts insights from data",
        "Statistics provides the foundation for ML"
    ]
    
    # Initialize comparison
    comparison = EmbeddingComparison()
    
    # Test with a subset of models for demo
    test_models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]
    
    print(f"Testing with {len(sample_texts)} sample texts")
    print(f"Models: {', '.join(test_models)}")
    
    # Run benchmark
    results_df = comparison.benchmark_all_models(sample_texts, test_models)
    
    if not results_df.empty:
        print("\n📊 Benchmark Results:")
        print(results_df[['model_name', 'throughput_texts_per_second', 'memory_used_mb', 'embedding_dimension']])
        
        # Create plots
        comparison.plot_performance_comparison(results_df)
        
        # Generate recommendation report
        report = comparison.create_recommendation_report(results_df)
        print(f"\n{report}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    demo_embedding_comparison()
