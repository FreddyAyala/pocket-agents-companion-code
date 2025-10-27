# Chapter 4: Essential Metrics for On-Device AI Performance

**On-Device AI: The Small Language Models Revolution**

This companion code demonstrates how to measure and optimize the key performance metrics for on-device AI systems, including TTFT, throughput, memory footprint, and energy efficiency.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to this directory
cd companion-code/chapters/chapter-04

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
jupyter notebook metrics_demo.ipynb
```

## 📋 What You'll Learn

- **TTFT (Time To First Token)**: Measuring and optimizing response latency
- **Throughput**: Tokens per second benchmarking and optimization
- **Memory Footprint**: RAM usage analysis and optimization strategies
- **Energy Efficiency**: Joules per token measurement for mobile devices
- **MVI (Minimum Viable Intelligence)**: Finding the right model size for your use case
- **Performance Profiling**: Identifying bottlenecks in your AI pipeline

## 🎯 Key Concepts

- **TTFT Threshold**: < 150ms for perceived instantaneity
- **Throughput Target**: 200+ tokens/second for fluid responses
- **Memory Limit**: ≤ 2GB for broad device compatibility
- **Energy Efficiency**: Joules/token for battery life optimization
- **MVI Testing**: Balancing capability with resource constraints

## 📊 Expected Results

- **TTFT**: 50-150ms on modern hardware
- **Throughput**: 100-500 tokens/second depending on model size
- **Memory**: 1-4GB for 3B-7B parameter models
- **Energy**: 0.1-1.0 Joules/token on mobile devices
- **MVI**: 3B+ parameters for complex reasoning tasks

## 🔬 Techniques Demonstrated

### 1. TTFT Measurement
- First token latency measurement
- Model loading time analysis
- Context window impact on TTFT
- Optimization strategies

### 2. Throughput Benchmarking
- Tokens per second calculation
- Batch processing optimization
- Memory bandwidth analysis
- Hardware utilization metrics

### 3. Memory Profiling
- RAM usage tracking
- Model size analysis
- KV cache optimization
- Memory leak detection

### 4. Energy Efficiency
- Power consumption measurement
- Thermal throttling detection
- Battery life estimation
- Mobile optimization strategies

### 5. MVI Testing
- Capability vs resource trade-offs
- Task-specific model selection
- Performance degradation analysis
- Quality threshold determination

## 🚀 Performance Benefits

- **Faster Response**: Sub-150ms TTFT for instant feel
- **Higher Throughput**: 200+ tokens/second for fluid generation
- **Lower Memory**: ≤ 2GB for broad compatibility
- **Better Battery Life**: Optimized energy consumption
- **Right-Sized Models**: MVI for your specific use case

## ⚖️ Trade-offs

- **Speed vs Quality**: Larger models are slower but more capable
- **Memory vs Performance**: More RAM enables larger models
- **Energy vs Speed**: Faster inference uses more power
- **Model Size vs Capability**: Smaller models are faster but less capable

## 🔗 Related Chapters

- Chapter 5: Model Compression - Quantization
- Chapter 7: Fine-Tuning & Adaptation
- Chapter 9: The Hardware Battlefield
- Chapter 12: The Hero Project (Capstone)

## 💡 Best Practices

1. **Start with TTFT**: Ensure < 150ms for good UX
2. **Measure Everything**: Track all key metrics
3. **Test on Target Hardware**: Use actual deployment devices
4. **Optimize Iteratively**: Make incremental improvements
5. **Consider Use Case**: Different apps have different requirements

## 🛠️ Files in this Chapter

- `metrics_demo.ipynb` - Comprehensive metrics demonstration
- `benchmark_runner.py` - Automated benchmarking tool
- `performance_profiler.py` - Advanced profiling utilities
- `energy_monitor.py` - Power consumption analysis

## 🎮 Interactive Demo

The Jupyter notebook provides a step-by-step walkthrough:

1. **TTFT Measurement**: Measure and optimize first token latency
2. **Throughput Analysis**: Benchmark tokens per second
3. **Memory Profiling**: Analyze RAM usage patterns
4. **Energy Monitoring**: Measure power consumption
5. **MVI Testing**: Find optimal model size
6. **Performance Optimization**: Apply optimization strategies

## 🔧 Troubleshooting

### Common Issues

- **"Out of memory"**: Reduce model size or batch size
- **"Slow inference"**: Check hardware acceleration
- **"High power usage"**: Optimize for mobile deployment
- **"Inconsistent performance"**: Check thermal throttling

### Performance Tips

1. **Use GPU acceleration** when available
2. **Optimize batch sizes** for your hardware
3. **Monitor thermal throttling** on mobile devices
4. **Profile memory usage** to find bottlenecks
5. **Test on target hardware** for accurate results

---

*This chapter provides the essential metrics and tools needed to build high-performance on-device AI systems.*
