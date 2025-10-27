#!/usr/bin/env python3
"""
Chapter 11: Edge Management - Concurrency & Parallel Processing

This module demonstrates thread-safe operations, TaskGroup-based concurrency,
and processor core optimization for edge AI deployment.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import threading
import queue
import time
import multiprocessing
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil

class ThreadSafeEngine:
    """
    Thread-safe inference engine for edge AI deployment.
    
    Implements atomic operations and decoupling to ensure
    safe concurrent access to model weights and KV cache.
    """
    
    def __init__(self, max_workers: int = None):
        """Initialize thread-safe engine"""
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.lock = threading.Lock()
        self.running = False
        self.workers = []
        
    def start(self):
        """Start the inference engine"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_thread, name=f"InferenceWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        print(f"🚀 Thread-safe engine started with {self.max_workers} workers")
    
    def stop(self):
        """Stop the inference engine"""
        self.running = False
        
        # Send shutdown signals
        for _ in self.workers:
            self.request_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
        self.thread_pool.shutdown(wait=True)
        print("⏹️ Thread-safe engine stopped")
    
    def _worker_thread(self):
        """Worker thread for processing requests"""
        while self.running:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=1)
                
                if request is None:  # Shutdown signal
                    break
                
                # Process request with thread safety
                response = self._process_request_safely(request)
                
                # Put response in queue
                self.response_queue.put(response)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Worker thread error: {e}")
    
    def _process_request_safely(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with thread safety"""
        with self.lock:
            # Simulate model inference (in production, this would be actual model)
            start_time = time.time()
            
            # Simulate processing time
            time.sleep(0.1)  # Simulate inference time
            
            result = {
                'id': request['id'],
                'result': f"Processed: {request['prompt'][:50]}...",
                'timestamp': time.time(),
                'processing_time': time.time() - start_time,
                'worker_id': threading.current_thread().name
            }
            
            return result
    
    def submit_request(self, request_id: str, prompt: str, **kwargs) -> None:
        """Submit request for processing"""
        request = {
            'id': request_id,
            'prompt': prompt,
            'timestamp': time.time(),
            **kwargs
        }
        
        self.request_queue.put(request)
    
    def get_response(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Get response from the queue"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'request_queue_size': self.request_queue.qsize(),
            'response_queue_size': self.response_queue.qsize(),
            'max_workers': self.max_workers,
            'active_workers': len([w for w in self.workers if w.is_alive()])
        }

class TaskGroupManager:
    """
    TaskGroup-based concurrency for RAG operations.
    
    Manages concurrent embedding generation and vector search
    while the main LLM processes user requests.
    """
    
    def __init__(self, max_concurrent: int = 4):
        """Initialize task group manager"""
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_tasks = {}
        self.task_results = {}
        
    async def run_concurrent_tasks(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Run multiple tasks concurrently"""
        
        async def run_single_task(task, task_id):
            async with self.semaphore:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task, *args, **kwargs)
                return task_id, result
        
        # Create tasks for all functions
        task_coroutines = [run_single_task(task, i) for i, task in enumerate(tasks)]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*task_coroutines)
        
        # Sort results by task ID
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    async def run_embedding_generation(self, texts: List[str], embedding_model=None) -> List[Any]:
        """Run embedding generation concurrently"""
        
        def generate_embedding(text):
            # Simulate embedding generation
            time.sleep(0.05)  # Simulate processing time
            return f"embedding_{hash(text) % 1000:03d}"
        
        # Split texts into batches
        batch_size = max(1, len(texts) // self.max_concurrent)
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Create tasks for each batch
        tasks = []
        for batch in batches:
            task = lambda b=batch: [generate_embedding(text) for text in b]
            tasks.append(task)
        
        # Run tasks concurrently
        results = await self.run_concurrent_tasks(tasks)
        
        # Flatten results
        embeddings = []
        for result in results:
            if result:
                embeddings.extend(result)
        
        return embeddings
    
    async def run_vector_search(self, query_embedding: str, vector_store=None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run vector search concurrently"""
        
        def search_vectors(embedding):
            # Simulate vector search
            time.sleep(0.03)  # Simulate search time
            return [
                {'id': f'doc_{i}', 'score': 0.9 - i*0.1, 'content': f'Document {i} content'}
                for i in range(top_k)
            ]
        
        # Create multiple search tasks
        tasks = [lambda: search_vectors(query_embedding) for _ in range(3)]  # 3 parallel searches
        
        # Run searches concurrently
        results = await self.run_concurrent_tasks(tasks)
        
        # Combine and deduplicate results
        all_results = []
        for result in results:
            all_results.extend(result)
        
        # Sort by score and remove duplicates
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result['id'])
        
        return unique_results[:top_k]

class ProcessorCoreOptimizer:
    """
    Processor core utilization optimizer for edge devices.
    
    Provides feedback loop for optimizing concurrent workers
    based on CPU usage patterns.
    """
    
    def __init__(self):
        """Initialize processor core optimizer"""
        self.cpu_count = multiprocessing.cpu_count()
        self.core_usage_history = []
        self.optimization_strategies = {
            'cpu_bound': self._optimize_for_cpu_bound,
            'io_bound': self._optimize_for_io_bound,
            'memory_bound': self._optimize_for_memory_bound
        }
    
    def monitor_core_usage(self) -> Dict[str, float]:
        """Monitor CPU core usage"""
        # Get per-core CPU usage
        per_core_usage = psutil.cpu_percent(percpu=True, interval=1)
        
        core_usage = {
            f'core_{i}': usage for i, usage in enumerate(per_core_usage)
        }
        
        # Store in history
        self.core_usage_history.append(core_usage)
        if len(self.core_usage_history) > 10:  # Keep last 10 measurements
            self.core_usage_history.pop(0)
        
        return core_usage
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current usage"""
        
        if not self.core_usage_history:
            self.monitor_core_usage()
        
        # Analyze core usage patterns
        latest_usage = self.core_usage_history[-1]
        avg_usage = sum(latest_usage.values()) / len(latest_usage)
        max_usage = max(latest_usage.values())
        min_usage = min(latest_usage.values())
        
        recommendations = []
        
        if max_usage - min_usage > 20:  # Uneven load distribution
            recommendations.append("Consider load balancing across cores")
        
        if avg_usage > 80:  # High CPU usage
            recommendations.append("Consider reducing number of workers")
        
        if avg_usage < 30:  # Low CPU usage
            recommendations.append("Consider increasing number of workers")
        
        if avg_usage > 90:  # Critical CPU usage
            recommendations.append("CRITICAL: Reduce workload immediately")
        
        return recommendations
    
    def optimize_for_workload(self, workload_type: str, model_size: int, available_memory: int) -> Dict[str, Any]:
        """Optimize processor usage for specific workload"""
        
        strategy = self.optimization_strategies.get(workload_type)
        if strategy:
            return strategy(model_size, available_memory)
        else:
            return self._default_optimization()
    
    def _optimize_for_cpu_bound(self, model_size: int, available_memory: int) -> Dict[str, Any]:
        """Optimize for CPU-bound workloads"""
        
        # Use all available cores
        num_workers = self.cpu_count
        
        # Optimize for parallel processing
        config = {
            'num_workers': num_workers,
            'batch_size': max(1, available_memory // (model_size * 2)),
            'threading_model': 'thread',
            'optimization_level': 'high',
            'reasoning': 'CPU-bound workload benefits from maximum parallelization'
        }
        
        return config
    
    def _optimize_for_io_bound(self, model_size: int, available_memory: int) -> Dict[str, Any]:
        """Optimize for I/O-bound workloads"""
        
        # Use more workers for I/O operations
        num_workers = self.cpu_count * 2
        
        config = {
            'num_workers': num_workers,
            'batch_size': 1,  # Process one at a time for I/O
            'threading_model': 'thread',
            'optimization_level': 'medium',
            'reasoning': 'I/O-bound workload benefits from more concurrent workers'
        }
        
        return config
    
    def _optimize_for_memory_bound(self, model_size: int, available_memory: int) -> Dict[str, Any]:
        """Optimize for memory-bound workloads"""
        
        # Use fewer workers to conserve memory
        num_workers = max(1, self.cpu_count // 2)
        
        # Smaller batch size to fit in memory
        batch_size = max(1, available_memory // (model_size * 4))
        
        config = {
            'num_workers': num_workers,
            'batch_size': batch_size,
            'threading_model': 'process',  # Use processes for memory isolation
            'optimization_level': 'low',
            'reasoning': 'Memory-bound workload requires conservative resource usage'
        }
        
        return config
    
    def _default_optimization(self) -> Dict[str, Any]:
        """Default optimization strategy"""
        return {
            'num_workers': self.cpu_count,
            'batch_size': 4,
            'threading_model': 'thread',
            'optimization_level': 'medium',
            'reasoning': 'Balanced approach for general workloads'
        }

def demonstrate_concurrency_management():
    """Demonstrate concurrency and parallel processing capabilities"""
    
    print("=" * 70)
    print("Chapter 11: Concurrency & Parallel Processing Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Demo 1: Thread-Safe Engine
    print("\n🔧 Demo 1: Thread-Safe Engine")
    print("-" * 40)
    
    engine = ThreadSafeEngine(max_workers=4)
    engine.start()
    
    # Submit multiple requests
    print("Submitting concurrent requests...")
    for i in range(8):
        engine.submit_request(f"req_{i}", f"Process this request {i}")
    
    # Collect responses
    responses = []
    for _ in range(8):
        response = engine.get_response(timeout=5)
        if response:
            responses.append(response)
            print(f"  ✅ {response['id']}: {response['result']} (Worker: {response['worker_id']})")
    
    # Show queue status
    status = engine.get_queue_status()
    print(f"\nQueue Status:")
    print(f"  Request Queue: {status['request_queue_size']}")
    print(f"  Response Queue: {status['response_queue_size']}")
    print(f"  Active Workers: {status['active_workers']}")
    
    engine.stop()
    
    # Demo 2: TaskGroup Concurrency
    print("\n\n🔄 Demo 2: TaskGroup Concurrency")
    print("-" * 40)
    
    async def run_taskgroup_demo():
        task_manager = TaskGroupManager(max_concurrent=3)
        
        # Test embedding generation
        print("Testing concurrent embedding generation...")
        texts = [f"Document {i} content" for i in range(10)]
        embeddings = await task_manager.run_embedding_generation(texts)
        print(f"  Generated {len(embeddings)} embeddings")
        
        # Test vector search
        print("Testing concurrent vector search...")
        search_results = await task_manager.run_vector_search("query_embedding", top_k=3)
        print(f"  Found {len(search_results)} results")
        for result in search_results:
            print(f"    {result['id']}: {result['score']:.2f}")
    
    # Run async demo
    asyncio.run(run_taskgroup_demo())
    
    # Demo 3: Processor Core Optimization
    print("\n\n⚡ Demo 3: Processor Core Optimization")
    print("-" * 40)
    
    optimizer = ProcessorCoreOptimizer()
    
    # Monitor core usage
    print("Monitoring CPU core usage...")
    core_usage = optimizer.monitor_core_usage()
    print(f"Core Usage: {core_usage}")
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print(f"Recommendations: {recommendations}")
    
    # Test different workload optimizations
    print("\nTesting workload optimizations:")
    
    test_scenarios = [
        ("cpu_bound", 1000, 4000),
        ("io_bound", 500, 2000),
        ("memory_bound", 2000, 1000)
    ]
    
    for workload_type, model_size, available_memory in test_scenarios:
        config = optimizer.optimize_for_workload(workload_type, model_size, available_memory)
        print(f"  {workload_type}: {config['num_workers']} workers, batch_size={config['batch_size']}")
        print(f"    Reasoning: {config['reasoning']}")
    
    print("\n💡 Key Takeaways:")
    print("  • Thread-safe operations prevent race conditions")
    print("  • TaskGroup concurrency maximizes throughput")
    print("  • Processor optimization adapts to workload type")
    print("  • Real-time monitoring enables dynamic adjustment")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_concurrency_management()
