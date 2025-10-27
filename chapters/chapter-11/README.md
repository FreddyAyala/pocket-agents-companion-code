# Chapter 11: Taming the Edge - Memory, Context, and Concurrency

This companion code demonstrates critical edge management techniques for production AI deployment on resource-constrained devices.

## Contents

### Core Examples
- `memory_management.py` - Device memory profiling, watermark system, and UMA support
- `context_optimization.py` - Three-tiered context compression system (500-token buffer, Summary Chain, Semantic Memory)
- `concurrency_management.py` - Thread-safe operations, TaskGroup concurrency, and processor core optimization
- `database_optimization.py` - Edge database optimization with WAL mode, connection pooling, and batch operations
- `production_patterns.py` - Error handling, graceful degradation, and state persistence systems
- `edge_management_demo.ipynb` - Interactive tutorial with all new features
- `test_edge_management.py` - Comprehensive test suite covering all functionality

### Setup & Documentation
- `requirements.txt` - Python dependencies
- `setup_and_test.sh` - Automated setup script
- `README.md` - This documentation

## Quick Start

### 1. Setup Environment
```bash
# Make setup script executable
chmod +x setup_and_test.sh

# Run setup
./setup_and_test.sh
```

### 2. Run Examples

#### Memory Management Demo
```bash
python memory_management.py
```

**What it demonstrates:**
- Device-specific memory profiling with UMA support
- Memory watermark system (LOW/MEDIUM/HIGH thresholds)
- DeviceMemoryManager with three non-negotiable boundaries
- Model loading capability assessment
- Memory pressure monitoring and degradation

#### Context Optimization Demo
```bash
python context_optimization.py
```

**What it demonstrates:**
- Three-tiered context compression system
- 500-token raw buffer (Tier 1)
- Summary Chain compression (Tier 2)
- Semantic Memory Injection (Tier 3)
- Dynamic context sizing and compression triggers

#### Concurrency Management Demo
```bash
python concurrency_management.py
```

**What it demonstrates:**
- Thread-safe LLM inference engine
- TaskGroup-based concurrent RAG operations
- Processor core utilization optimization
- Workload-specific performance tuning

#### Database Optimization Demo
```bash
python database_optimization.py
```

**What it demonstrates:**
- SQLite WAL mode optimization
- Connection pooling for edge devices
- Batch operations for embeddings and documents
- Database performance monitoring

#### Production Patterns Demo
```bash
python production_patterns.py
```

**What it demonstrates:**
- Production-grade error handling with severity levels
- Graceful degradation management
- Agent state persistence across reboots
- Self-healing system architecture

### 3. Run Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest test_edge_management.py -v

# Run with coverage
pytest test_edge_management.py --cov=. --cov-report=html
```

## What This Code Demonstrates

### Memory Management (`memory_management.py`)

#### Device Profiling with UMA Support
```python
from memory_management import DeviceMemoryProfiler, DeviceMemoryManager

# Profile current device
profiler = DeviceMemoryProfiler()
profiler.print_profile()

# Get structured profile with UMA detection
profile = profiler.get_profile()
print(f"Total Memory: {profile.total_memory_mb:.1f} MB")
print(f"Available: {profile.available_memory_mb:.1f} MB")
print(f"UMA Architecture: {profile.is_uma}")
```

#### Memory Watermark System
```python
from memory_management import DeviceMemoryManager, WatermarkConfig

# Initialize memory manager with watermark protection
config = WatermarkConfig()
manager = DeviceMemoryManager(profiler, config)

# Check model loading capability
model_size = 2000  # MB
can_load = manager.can_load_model(model_size)
print(f"Can load {model_size}MB model: {can_load}")

# Get optimal context size
context_size = manager.get_optimal_context_size(model_size)
print(f"Optimal context size: {context_size} tokens")
```

#### Memory Allocation Strategy

The DeviceMemoryManager uses three non-negotiable boundaries:
- **Model Weights Size**: Fixed memory for quantized SLM
- **Safety Margin**: 5-10% buffer for OS and system processes
- **KV Cache Budget**: Remaining memory for dynamic operations

This ensures stable performance without OOM errors.

### Context Optimization (`context_optimization.py`)

#### Three-Tiered Context System

```python
from context_optimization import ThreeTieredContextSystem, ContextConfig

# Configure three-tiered system
config = ContextConfig(
    max_context_tokens=2048,
    buffer_tokens=500,           # Tier 1: Raw Buffer
    system_prompt_tokens=100,
    compression_threshold=0.9
)

system = ThreeTieredContextSystem(config)

# Add messages to the system
system.add_message("user", "Hello, I need help with Python", 10)
system.add_message("assistant", "I'd be happy to help with Python!", 12)

# Get final prompt with semantic memory
prompt = system.get_final_prompt("How do I create a list?")
```

**Three-Tiered Architecture:**
- **Tier 1: Raw Buffer** (~500 tokens) - Immediate conversation history
- **Tier 2: Summary Chain** - Compressed long-term sequential memory
- **Tier 3: Semantic Memory** - Non-sequential recall with embeddings

#### Dynamic Context Sizing

```python
from context_optimization import DynamicContextSizer

# Calculate optimal context for device
optimal_size, reasoning = DynamicContextSizer.calculate_optimal_context_size(
    available_memory_mb=4000,
    model_size_mb=1000,
    desired_size=4096
)

print(f"Optimal Context: {optimal_size} tokens")
print(f"Reasoning: {reasoning}")
```

### Concurrency Management (`concurrency_management.py`)

#### Thread-Safe LLM Engine
```python
from concurrency_management import ThreadSafeEngine

# Initialize thread-safe engine
engine = ThreadSafeEngine(max_workers=4)
engine.start()

# Submit concurrent requests
engine.submit_request("req_1", "What is AI?", max_tokens=100)
engine.submit_request("req_2", "Explain machine learning", max_tokens=150)

# Get responses
response1 = engine.get_response(timeout=5)
response2 = engine.get_response(timeout=5)
```

#### TaskGroup Concurrency for RAG
```python
from concurrency_management import TaskGroupManager

# Initialize task group manager
task_manager = TaskGroupManager(max_concurrent=3)

# Run concurrent embedding generation
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = await task_manager.run_embedding_generation(texts)

# Run concurrent vector search
results = await task_manager.run_vector_search("query", top_k=5)
```

### Database Optimization (`database_optimization.py`)

#### Edge Database Optimizer
```python
from database_optimization import EdgeDatabaseOptimizer, DatabaseConfig

# Configure optimized database
config = DatabaseConfig(
    db_path="edge_rag.db",
    connection_pool_size=5,
    enable_wal_mode=True
)

db_optimizer = EdgeDatabaseOptimizer(config)

# Batch insert embeddings
embeddings = [
    {'vector_id': f'embed_{i}', 'vector_data': f'data_{i}'.encode(), 'metadata': {}}
    for i in range(1000)
]
inserted = db_optimizer.batch_insert_embeddings(embeddings)
```

### Production Patterns (`production_patterns.py`)

#### Error Handling and Degradation
```python
from production_patterns import ProductionErrorHandler, GracefulDegradationManager, ErrorSeverity

# Initialize production systems
error_handler = ProductionErrorHandler()
degradation_manager = GracefulDegradationManager(error_handler)

# Handle errors with severity levels
try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    error_handler.handle_error(e, {'operation': 'risky_op'}, ErrorSeverity.HIGH)

# Check degradation status
status = degradation_manager.get_degradation_status()
print(f"Current degradation level: {status['degradation_level']}")
```

#### State Persistence
```python
from production_patterns import StatePersistenceManager

# Initialize state manager
state_manager = StatePersistenceManager("agent_state.json")

# Save agent state
state_manager.set_state("current_goal", "Plan a trip to Mars")
state_manager.set_state("conversation_summary", "User wants to travel to Mars")
state_manager.save_state()

# Load state after restart
state_manager.load_state()
goal = state_manager.get_state("current_goal")
```

## Key Concepts

### Memory Management Strategies

1. **Device-Specific Profiling with UMA Support**
   - Detect total and available memory
   - Identify GPU/NPU capabilities and UMA architecture
   - Calculate safe allocation limits with three boundaries

2. **Memory Watermark Protection System**
   - LOW (70%): Minor cleanup and GC
   - MEDIUM (85%): Reduce batch size, clear caches
   - HIGH (95%): Aggressive context truncation
   - Monitor memory pressure continuously

3. **Three Non-Negotiable Boundaries**
   - Model Weights Size: Fixed memory for quantized SLM
   - Safety Margin: 5-10% buffer for OS and system processes
   - KV Cache Budget: Remaining memory for dynamic operations

### Context Window Optimization

1. **Three-Tiered Context System**
   - **Tier 1**: Raw Buffer (~500 tokens) for immediate coherence
   - **Tier 2**: Summary Chain for compressed long-term memory
   - **Tier 3**: Semantic Memory with embeddings for non-sequential recall

2. **Dynamic Context Sizing**
   - Adapts to available memory and device capabilities
   - Scales from 512 to 32K tokens
   - Balances capability with constraints

3. **Compression Triggers**
   - Automatic compression at 90% capacity
   - Maintains conversation quality while managing memory
   - Ensures minimum conversation turn preserved

### Concurrency & Parallel Processing

1. **Thread-Safe Operations**
   - Atomic operations for LLM inference
   - Decoupling of request/response processing
   - Protection of model weights and KV cache

2. **TaskGroup-Based Concurrency**
   - Concurrent embedding generation
   - Parallel vector search operations
   - Optimized for RAG workloads

3. **Processor Core Optimization**
   - CPU-bound: Use all cores with threading
   - I/O-bound: 2x cores with async operations
   - Memory-bound: Reduce workers to prevent thrashing

### Database Performance Tuning

1. **SQLite Optimization**
   - WAL mode for concurrent reads/writes
   - Connection pooling for edge devices
   - Batch operations for efficient data ingestion

2. **Edge-Specific Configurations**
   - Optimized PRAGMA settings
   - Memory-mapped I/O
   - Incremental vacuum for maintenance

### Production Architecture Patterns

1. **Error Handling**
   - Severity levels: LOW, MEDIUM, HIGH, CRITICAL
   - Retry logic with exponential backoff
   - Structured error responses

2. **Graceful Degradation**
   - Performance vs. capability trade-offs
   - Automatic system adjustment under stress
   - Maintains core functionality

3. **State Persistence**
   - Agent state survival across reboots
   - Background processing optimization
   - Long-term autonomy support

## Testing

The comprehensive test suite covers:

### Memory Management Tests
- Device memory profiling with UMA support
- Memory watermark system functionality
- DeviceMemoryManager boundary calculations
- Model loading capability assessment
- Memory pressure monitoring

### Context Optimization Tests
- Three-tiered context system functionality
- Raw buffer management (Tier 1)
- Summary chain compression (Tier 2)
- Semantic memory injection (Tier 3)
- Dynamic context sizing

### Concurrency Management Tests
- Thread-safe engine operations
- TaskGroup concurrent execution
- Processor core optimization
- Workload-specific tuning

### Database Optimization Tests
- SQLite WAL mode configuration
- Connection pooling functionality
- Batch operations for embeddings and documents
- Database performance monitoring

### Production Patterns Tests
- Error handling with severity levels
- Graceful degradation management
- State persistence operations
- Integration scenarios

Run tests to verify your environment:
```bash
pytest test_edge_management.py -v
```

## Troubleshooting

### Common Issues

1. **Import Error: psutil not found**
   ```bash
   pip install psutil
   ```

2. **PyTorch not available**
   ```bash
   pip install torch
   ```

3. **Tests failing**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

## Production Recommendations

### For Mobile Devices
- Use Q4_K_M quantization
- Limit context to 2K-4K tokens
- Set compression threshold to 0.85
- Monitor memory continuously

### For Tablets
- Use Q8_0 or Q4_K_M quantization
- Context up to 4K-8K tokens
- Set compression threshold to 0.9
- Balance performance with quality

### For Desktop/Server
- Use FP16 or Q8_0 quantization
- Context up to 8K-16K tokens
- Set compression threshold to 0.9
- Optimize for throughput

## Next Steps

After mastering edge management:
- Explore Chapter 12: Agentic Best Practices
- Implement Hero Project with edge optimizations
- Deploy to production environments
- Monitor and optimize continuously

---

*"Efficient edge management is the difference between a prototype and a production system."*
