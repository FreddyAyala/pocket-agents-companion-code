#!/usr/bin/env python3
"""
Chapter 11: Edge Management - Test Suite

Tests for memory management and context optimization modules.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import pytest
import time
import threading
import asyncio
import os
import tempfile
from memory_management import DeviceMemoryProfiler, MemoryProfile, DeviceMemoryManager, WatermarkConfig, WatermarkLevel
from context_optimization import ContextWindowManager, ContextConfig, DynamicContextSizer, ThreeTieredContextSystem
from concurrency_management import ThreadSafeEngine, TaskGroupManager, ProcessorCoreOptimizer
from database_optimization import DatabaseConfig, EdgeDatabaseOptimizer, BatchProcessor
from production_patterns import ProductionErrorHandler, GracefulDegradationManager, StatePersistenceManager, ErrorSeverity

class TestDeviceMemoryProfiler:
    """Tests for Device Memory Profiler"""
    
    def test_profiler_initialization(self):
        """Test profiler initializes correctly"""
        profiler = DeviceMemoryProfiler()
        assert profiler.device_info is not None
        assert profiler.memory_limits is not None
    
    def test_memory_profile(self):
        """Test memory profile generation"""
        profiler = DeviceMemoryProfiler()
        profile = profiler.get_profile()
        
        assert isinstance(profile, MemoryProfile)
        assert profile.total_memory_mb > 0
        assert profile.available_memory_mb > 0
        assert profile.cpu_count > 0
    
    def test_memory_recommendations(self):
        """Test memory recommendations"""
        profiler = DeviceMemoryProfiler()
        recommendations = profiler.get_memory_recommendations(1000)  # 1GB model
        
        assert 'max_context_length' in recommendations
        assert 'optimal_batch_size' in recommendations
        assert 'recommended_quantization' in recommendations
        assert recommendations['max_context_length'] > 0
        assert recommendations['optimal_batch_size'] >= 1
    
    def test_quantization_recommendation(self):
        """Test quantization recommendations"""
        profiler = DeviceMemoryProfiler()
        
        # Test with small model
        small_model_rec = profiler.get_memory_recommendations(500)
        assert 'fp16' in small_model_rec['recommended_quantization'].lower() or \
               'q8' in small_model_rec['recommended_quantization'].lower()
        
        # Test with large model (assuming device has limited memory)
        large_model_rec = profiler.get_memory_recommendations(10000)
        # Should recommend quantization for very large models
        assert 'q4' in large_model_rec['recommended_quantization'].lower() or \
               large_model_rec['can_load_model'] == False

class TestContextWindowManager:
    """Tests for Context Window Manager"""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        config = ContextConfig(max_context_tokens=2048, buffer_tokens=500)
        manager = ContextWindowManager(config)
        
        assert manager.token_count == 0
        assert manager.total_messages == 0
        assert len(manager.conversation_history) == 0
    
    def test_add_message(self):
        """Test adding messages"""
        config = ContextConfig(max_context_tokens=2048)
        manager = ContextWindowManager(config)
        
        manager.add_message("user", "Hello", 5)
        assert manager.token_count == 5
        assert manager.total_messages == 1
        assert len(manager.conversation_history) == 1
    
    def test_context_utilization(self):
        """Test context utilization calculation"""
        config = ContextConfig(max_context_tokens=1000)
        manager = ContextWindowManager(config)
        
        manager.add_message("user", "Test message", 100)
        assert manager.utilization == 0.1  # 100/1000
    
    def test_context_compression(self):
        """Test context compression triggers"""
        config = ContextConfig(
            max_context_tokens=1000,
            buffer_tokens=100,
            compression_threshold=0.9
        )
        manager = ContextWindowManager(config)
        
        # Add messages until compression threshold
        for i in range(10):
            manager.add_message("user", f"Message {i}", 100)
        
        # Should have triggered compression
        assert manager.compressions_performed > 0
        assert manager.utilization < 1.0
    
    def test_available_tokens(self):
        """Test available tokens calculation"""
        config = ContextConfig(
            max_context_tokens=2048,
            buffer_tokens=500,
            system_prompt_tokens=100
        )
        manager = ContextWindowManager(config)
        
        expected_available = 2048 - 500 - 100  # max - buffer - system
        assert manager.available_tokens == expected_available
        
        # Add some messages
        manager.add_message("user", "Test", 50)
        assert manager.available_tokens == expected_available - 50
    
    def test_get_statistics(self):
        """Test statistics generation"""
        config = ContextConfig(max_context_tokens=2048)
        manager = ContextWindowManager(config)
        
        manager.add_message("user", "Hello", 10)
        stats = manager.get_statistics()
        
        assert 'max_context_tokens' in stats
        assert 'current_tokens' in stats
        assert 'utilization' in stats
        assert stats['current_tokens'] == 10
        assert stats['total_messages'] == 1

class TestDynamicContextSizer:
    """Tests for Dynamic Context Sizer"""
    
    def test_context_sizing_comfortable(self):
        """Test context sizing with comfortable memory"""
        optimal_size, reasoning = DynamicContextSizer.calculate_optimal_context_size(
            available_memory_mb=8000,
            model_size_mb=2000,
            desired_size=4096
        )
        
        assert optimal_size > 0
        assert optimal_size <= 4096
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
    
    def test_context_sizing_constrained(self):
        """Test context sizing with constrained memory"""
        optimal_size, reasoning = DynamicContextSizer.calculate_optimal_context_size(
            available_memory_mb=2000,
            model_size_mb=1500,
            desired_size=4096
        )
        
        assert optimal_size > 0
        # Should be smaller than desired due to memory constraints
        assert optimal_size <= 4096
        assert 'constrained' in reasoning.lower() or 'adequate' in reasoning.lower()
    
    def test_context_sizing_power_of_two(self):
        """Test that context sizes are powers of 2"""
        valid_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        
        optimal_size, _ = DynamicContextSizer.calculate_optimal_context_size(
            available_memory_mb=4000,
            model_size_mb=1000,
            desired_size=4096
        )
        
        assert optimal_size in valid_sizes

class TestDeviceMemoryManager:
    """Tests for Device Memory Manager with Watermark system"""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initializes correctly"""
        profiler = DeviceMemoryProfiler()
        config = WatermarkConfig()
        manager = DeviceMemoryManager(profiler, config)
        
        assert manager.profiler is not None
        assert manager.config is not None
        assert manager.model_memory_limit > 0
        assert manager.safety_margin > 0
        assert manager.kv_cache_budget > 0
    
    def test_memory_boundaries_calculation(self):
        """Test memory boundaries are calculated correctly"""
        profiler = DeviceMemoryProfiler()
        manager = DeviceMemoryManager(profiler)
        
        # Safety margin should be 10% of total memory
        total_memory = profiler.device_info['total_memory']
        expected_safety = int(total_memory * 0.1)
        assert manager.safety_margin == expected_safety
        
        # Model memory should be 40% of available after safety
        available_after_safety = total_memory - manager.safety_margin
        expected_model = int(available_after_safety * 0.4)
        assert manager.model_memory_limit == expected_model
    
    def test_can_load_model(self):
        """Test model loading capability check"""
        profiler = DeviceMemoryProfiler()
        manager = DeviceMemoryManager(profiler)
        
        # Small model should be loadable
        small_model_size = manager.model_memory_limit // 2
        assert manager.can_load_model(small_model_size) == True
        
        # Large model should not be loadable
        large_model_size = manager.model_memory_limit * 2
        assert manager.can_load_model(large_model_size) == False
    
    def test_optimal_context_size(self):
        """Test optimal context size calculation"""
        profiler = DeviceMemoryProfiler()
        manager = DeviceMemoryManager(profiler)
        
        # Test with loadable model
        model_size = manager.model_memory_limit // 2
        context_size = manager.get_optimal_context_size(model_size)
        assert context_size > 0
        
        # Test with unloadable model
        large_model_size = manager.model_memory_limit * 2
        context_size = manager.get_optimal_context_size(large_model_size)
        assert context_size == 0
    
    def test_watermark_monitoring(self):
        """Test watermark monitoring system"""
        profiler = DeviceMemoryProfiler()
        manager = DeviceMemoryManager(profiler)
        
        # Test watermark level detection
        low_usage = 0.5
        medium_usage = 0.8
        high_usage = 0.96
        
        assert manager._get_watermark_level(low_usage) == WatermarkLevel.LOW
        assert manager._get_watermark_level(medium_usage) == WatermarkLevel.MEDIUM
        assert manager._get_watermark_level(high_usage) == WatermarkLevel.HIGH

class TestThreeTieredContextSystem:
    """Tests for Three-Tiered Context System"""
    
    def test_system_initialization(self):
        """Test three-tiered system initializes correctly"""
        config = ContextConfig(max_context_tokens=2048, buffer_tokens=500)
        system = ThreeTieredContextSystem(config)
        
        assert system.config is not None
        assert len(system.raw_buffer) == 0
        assert len(system.summary_chain) == 0
        assert len(system.semantic_index) == 0
        assert system.compression_count == 0
    
    def test_add_message(self):
        """Test adding messages to the system"""
        config = ContextConfig(max_context_tokens=2048, buffer_tokens=500)
        system = ThreeTieredContextSystem(config)
        
        system.add_message("user", "Hello world", 10)
        
        assert len(system.raw_buffer) == 1
        assert system.raw_buffer[0]['role'] == "user"
        assert system.raw_buffer[0]['content'] == "Hello world"
        assert system.raw_buffer[0]['tokens'] == 10
    
    def test_compression_trigger(self):
        """Test compression is triggered at threshold"""
        config = ContextConfig(
            max_context_tokens=1000,
            buffer_tokens=100,
            compression_threshold=0.9
        )
        system = ThreeTieredContextSystem(config)
        
        # Add messages until compression threshold
        for i in range(20):
            system.add_message("user", f"Message {i}", 50)
        
        # Should have triggered compression
        assert system.compression_count > 0
    
    def test_semantic_memory_retrieval(self):
        """Test semantic memory retrieval"""
        config = ContextConfig(max_context_tokens=2048, buffer_tokens=500)
        system = ThreeTieredContextSystem(config)
        
        # Add messages with keywords
        system.add_message("user", "I need help with Python programming", 10)
        system.add_message("assistant", "Python is great for data science and web development", 15)
        
        # Test semantic retrieval
        matches = system.get_semantic_matches("Python programming", top_k=3)
        assert isinstance(matches, list)
        assert len(matches) <= 3
    
    def test_final_prompt_construction(self):
        """Test final prompt construction"""
        config = ContextConfig(max_context_tokens=2048, buffer_tokens=500)
        system = ThreeTieredContextSystem(config)
        
        # Add some messages
        system.add_message("user", "Hello", 5)
        system.add_message("assistant", "Hi there!", 8)
        
        # Get final prompt
        prompt = system.get_final_prompt("How are you?")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Hello" in prompt
        assert "How are you?" in prompt

class TestThreadSafeEngine:
    """Tests for Thread-Safe Engine"""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = ThreadSafeEngine(max_workers=2)
        
        assert engine.max_workers == 2
        assert not engine.running
        assert len(engine.workers) == 0
    
    def test_engine_start_stop(self):
        """Test engine start and stop"""
        engine = ThreadSafeEngine(max_workers=2)
        
        # Start engine
        engine.start()
        assert engine.running
        assert len(engine.workers) == 2
        
        # Stop engine
        engine.stop()
        assert not engine.running
    
    def test_request_processing(self):
        """Test request processing"""
        engine = ThreadSafeEngine(max_workers=2)
        engine.start()
        
        try:
            # Submit request
            engine.submit_request("test_1", "Test request")
            
            # Get response
            response = engine.get_response(timeout=5)
            assert response is not None
            assert response['id'] == "test_1"
            assert "Test request" in response['result']
        finally:
            engine.stop()
    
    def test_queue_status(self):
        """Test queue status reporting"""
        engine = ThreadSafeEngine(max_workers=2)
        
        status = engine.get_queue_status()
        assert 'request_queue_size' in status
        assert 'response_queue_size' in status
        assert 'max_workers' in status
        assert 'active_workers' in status

class TestTaskGroupManager:
    """Tests for TaskGroup Manager"""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        manager = TaskGroupManager(max_concurrent=3)
        
        assert manager.max_concurrent == 3
        assert manager.semaphore._value == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_tasks(self):
        """Test concurrent task execution"""
        manager = TaskGroupManager(max_concurrent=2)
        
        def test_task(x):
            time.sleep(0.1)
            return x * 2
        
        tasks = [test_task for _ in range(3)]
        results = await manager.run_concurrent_tasks(tasks, 5)
        
        assert len(results) == 3
        assert all(result == 10 for result in results)
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test concurrent embedding generation"""
        manager = TaskGroupManager(max_concurrent=2)
        
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = await manager.run_embedding_generation(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, str) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_vector_search(self):
        """Test concurrent vector search"""
        manager = TaskGroupManager(max_concurrent=2)
        
        results = await manager.run_vector_search("query", top_k=3)
        
        assert len(results) == 3
        assert all('id' in result for result in results)
        assert all('score' in result for result in results)

class TestProcessorCoreOptimizer:
    """Tests for Processor Core Optimizer"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly"""
        optimizer = ProcessorCoreOptimizer()
        
        assert optimizer.cpu_count > 0
        assert len(optimizer.optimization_strategies) == 3
    
    def test_core_usage_monitoring(self):
        """Test core usage monitoring"""
        optimizer = ProcessorCoreOptimizer()
        
        usage = optimizer.monitor_core_usage()
        assert isinstance(usage, dict)
        assert len(usage) == optimizer.cpu_count
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        optimizer = ProcessorCoreOptimizer()
        
        recommendations = optimizer.get_optimization_recommendations()
        assert isinstance(recommendations, list)
    
    def test_workload_optimization(self):
        """Test workload-specific optimization"""
        optimizer = ProcessorCoreOptimizer()
        
        # Test CPU-bound optimization
        config = optimizer.optimize_for_workload("cpu_bound", 1000, 4000)
        assert config['num_workers'] == optimizer.cpu_count
        assert config['threading_model'] == 'thread'
        
        # Test I/O-bound optimization
        config = optimizer.optimize_for_workload("io_bound", 500, 2000)
        assert config['num_workers'] == optimizer.cpu_count * 2
        
        # Test memory-bound optimization
        config = optimizer.optimize_for_workload("memory_bound", 2000, 1000)
        assert config['num_workers'] <= optimizer.cpu_count

class TestEdgeDatabaseOptimizer:
    """Tests for Edge Database Optimizer"""
    
    def test_database_initialization(self):
        """Test database optimizer initializes correctly"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            config = DatabaseConfig(db_path=db_path)
            db_optimizer = EdgeDatabaseOptimizer(config)
            
            assert db_optimizer.config is not None
            assert db_optimizer.connection_count > 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
    
    def test_batch_insert_embeddings(self):
        """Test batch embedding insertion"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            config = DatabaseConfig(db_path=db_path)
            db_optimizer = EdgeDatabaseOptimizer(config)
            
            embeddings = [
                {'vector_id': f'embed_{i}', 'vector_data': f'data_{i}'.encode(), 'metadata': {}}
                for i in range(10)
            ]
            
            inserted = db_optimizer.batch_insert_embeddings(embeddings)
            assert inserted == 10
            
            # Verify insertion
            results = db_optimizer.search_embeddings('embed_', limit=5)
            assert len(results) == 5
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
    
    def test_batch_insert_documents(self):
        """Test batch document insertion"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            config = DatabaseConfig(db_path=db_path)
            db_optimizer = EdgeDatabaseOptimizer(config)
            
            documents = [
                {'title': f'Doc {i}', 'content': f'Content {i}', 'type': 'text', 'metadata': {}}
                for i in range(5)
            ]
            
            inserted = db_optimizer.batch_insert_documents(documents)
            assert inserted == 5
            
            # Verify insertion
            results = db_optimizer.search_documents('Doc', limit=3)
            assert len(results) == 3
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

class TestProductionErrorHandler:
    """Tests for Production Error Handler"""
    
    def test_handler_initialization(self):
        """Test error handler initializes correctly"""
        handler = ProductionErrorHandler()
        
        assert handler.logger is not None
        assert len(handler.degradation_strategies) == 4
        assert handler.max_retries == 3
    
    def test_error_handling(self):
        """Test error handling with different severities"""
        handler = ProductionErrorHandler()
        
        # Test low severity error
        try:
            raise ValueError("Test error")
        except Exception as e:
            handler.handle_error(e, {'operation': 'test'}, ErrorSeverity.LOW)
        
        # Test high severity error
        try:
            raise MemoryError("Out of memory")
        except Exception as e:
            handler.handle_error(e, {'operation': 'test'}, ErrorSeverity.HIGH)
        
        # Check error counts
        stats = handler.get_error_stats()
        assert stats['total_errors'] == 2
        assert 'ValueError' in stats['error_counts']
        assert 'MemoryError' in stats['error_counts']
    
    def test_retry_logic(self):
        """Test retry logic"""
        handler = ProductionErrorHandler()
        
        # Test should retry
        context = {'retry_count': 0}
        assert handler._should_retry(ValueError("Test"), context) == True
        
        # Test should not retry (too many retries)
        context = {'retry_count': 5}
        assert handler._should_retry(ValueError("Test"), context) == False
        
        # Test should not retry (non-retryable error)
        context = {'retry_count': 0}
        assert handler._should_retry(AttributeError("Test"), context) == False

class TestGracefulDegradationManager:
    """Tests for Graceful Degradation Manager"""
    
    def test_degradation_manager_initialization(self):
        """Test degradation manager initializes correctly"""
        error_handler = ProductionErrorHandler()
        manager = GracefulDegradationManager(error_handler)
        
        assert manager.error_handler is not None
        assert manager.degradation_level == 0
        assert manager.max_degradation_level == 3
    
    def test_degradation_increase(self):
        """Test degradation level increase"""
        error_handler = ProductionErrorHandler()
        manager = GracefulDegradationManager(error_handler)
        
        assert manager.degradation_level == 0
        
        manager.increase_degradation()
        assert manager.degradation_level == 1
        
        manager.increase_degradation()
        assert manager.degradation_level == 2
        
        # Should not exceed max
        manager.increase_degradation()
        manager.increase_degradation()
        assert manager.degradation_level == 3
    
    def test_degradation_decrease(self):
        """Test degradation level decrease"""
        error_handler = ProductionErrorHandler()
        manager = GracefulDegradationManager(error_handler)
        
        # Increase first
        manager.increase_degradation()
        manager.increase_degradation()
        assert manager.degradation_level == 2
        
        # Decrease
        manager.decrease_degradation()
        assert manager.degradation_level == 1
        
        manager.decrease_degradation()
        assert manager.degradation_level == 0
        
        # Should not go below 0
        manager.decrease_degradation()
        assert manager.degradation_level == 0
    
    def test_degradation_status(self):
        """Test degradation status reporting"""
        error_handler = ProductionErrorHandler()
        manager = GracefulDegradationManager(error_handler)
        
        status = manager.get_degradation_status()
        assert 'degradation_level' in status
        assert 'max_degradation_level' in status
        assert 'status' in status
        assert status['degradation_level'] == 0
        assert status['status'] == 'normal'

class TestStatePersistenceManager:
    """Tests for State Persistence Manager"""
    
    def test_state_manager_initialization(self):
        """Test state manager initializes correctly"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            storage_path = tmp.name
        
        try:
            manager = StatePersistenceManager(storage_path)
            
            assert manager.storage_path == storage_path
            assert len(manager.state) == 0
            assert manager.auto_save_interval == 30
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)
    
    def test_state_operations(self):
        """Test state set/get operations"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            storage_path = tmp.name
        
        try:
            manager = StatePersistenceManager(storage_path)
            
            # Set state
            manager.set_state("key1", "value1")
            manager.set_state("key2", {"nested": "value"})
            
            # Get state
            assert manager.get_state("key1") == "value1"
            assert manager.get_state("key2") == {"nested": "value"}
            assert manager.get_state("nonexistent", "default") == "default"
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)
    
    def test_state_persistence(self):
        """Test state save/load operations"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            storage_path = tmp.name
        
        try:
            # Create manager and set state
            manager1 = StatePersistenceManager(storage_path)
            manager1.set_state("test_key", "test_value")
            manager1.set_state("number", 42)
            manager1.save_state()
            
            # Create new manager and load state
            manager2 = StatePersistenceManager(storage_path)
            manager2.load_state()
            
            assert manager2.get_state("test_key") == "test_value"
            assert manager2.get_state("number") == 42
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)
    
    def test_state_summary(self):
        """Test state summary generation"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            storage_path = tmp.name
        
        try:
            manager = StatePersistenceManager(storage_path)
            manager.set_state("key1", "value1")
            manager.set_state("key2", "value2")
            
            summary = manager.get_state_summary()
            assert 'keys' in summary
            assert 'size' in summary
            assert 'storage_path' in summary
            assert len(summary['keys']) == 2
            assert summary['size'] == 2
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)

def test_integration_memory_and_context():
    """Integration test: Memory profiling informs context sizing"""
    profiler = DeviceMemoryProfiler()
    profile = profiler.get_profile()
    
    # Use profiler recommendations to configure context manager
    model_size_mb = 2000
    recommendations = profiler.get_memory_recommendations(model_size_mb)
    
    # Create context config based on recommendations
    config = ContextConfig(
        max_context_tokens=recommendations['max_context_length'],
        buffer_tokens=500
    )
    manager = ContextWindowManager(config)
    
    # Verify integration works
    assert manager.config.max_context_tokens > 0
    assert manager.available_tokens > 0
    assert manager.utilization == 0.0

def test_integration_complete_system():
    """Integration test: Complete edge management system"""
    # Test complete system integration
    profiler = DeviceMemoryProfiler()
    memory_manager = DeviceMemoryManager(profiler)
    
    # Test memory manager
    assert memory_manager.model_memory_limit > 0
    
    # Test context system
    config = ContextConfig(max_context_tokens=2048, buffer_tokens=500)
    context_system = ThreeTieredContextSystem(config)
    context_system.add_message("user", "Test message", 10)
    assert len(context_system.raw_buffer) == 1
    
    # Test production systems
    error_handler = ProductionErrorHandler()
    degradation_manager = GracefulDegradationManager(error_handler)
    state_manager = StatePersistenceManager("test_state.json")
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'test'}, ErrorSeverity.LOW)
    
    # Test degradation
    degradation_manager.increase_degradation()
    assert degradation_manager.degradation_level == 1
    
    # Test state persistence
    state_manager.set_state("test", "value")
    state_manager.save_state()
    
    # Cleanup
    if os.path.exists("test_state.json"):
        os.remove("test_state.json")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

