#!/usr/bin/env python3
"""
Chapter 11: Edge Management - Memory Management

This module demonstrates device-specific memory profiling and optimization
strategies for edge AI deployment, including the DeviceMemoryManager with
Watermark system and Unified Memory Architecture (UMA) support.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import psutil
import platform
import sys
import threading
import time
import gc
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. GPU detection disabled.")

class WatermarkLevel(Enum):
    """Memory watermark levels for degradation triggers"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DegradationAction(Enum):
    """Actions to take at different watermark levels"""
    NONE = "none"
    CLEANUP = "cleanup"
    REDUCE_BATCH = "reduce_batch"
    REDUCE_CONTEXT = "reduce_context"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

@dataclass
class MemoryProfile:
    """Memory profile for a device"""
    platform: str
    architecture: str
    processor: str
    total_memory_mb: float
    available_memory_mb: float
    cpu_count: int
    cpu_freq_mhz: Optional[float]
    has_gpu: bool
    gpu_memory_mb: Optional[float]
    is_uma: bool = False  # Unified Memory Architecture

@dataclass
class WatermarkConfig:
    """Configuration for memory watermarks"""
    low_threshold: float = 0.70    # 70%
    medium_threshold: float = 0.85  # 85%
    high_threshold: float = 0.95   # 95%
    check_interval: float = 1.0    # seconds
    safety_margin: float = 0.10    # 10% safety margin

class DeviceMemoryProfiler:
    """
    Device memory profiler for edge AI deployment.
    
    Identifies device-specific limits and provides memory optimization
    recommendations for model deployment.
    """
    
    def __init__(self):
        """Initialize device memory profiler"""
        self.device_info = self._get_device_info()
        self.memory_limits = self._calculate_memory_limits()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        
        vm = psutil.virtual_memory()
        
        info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'total_memory': vm.total,
            'available_memory': vm.available,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # Add GPU information if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory'] = []
            for i in range(torch.cuda.device_count()):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                info['gpu_memory'].append(gpu_mem)
        elif TORCH_AVAILABLE and torch.backends.mps.is_available():
            info['gpu_type'] = 'Apple Silicon (MPS)'
            info['gpu_memory'] = ['Unified Memory Architecture']
        else:
            info['gpu_count'] = 0
            info['gpu_memory'] = []
        
        return info
    
    def _calculate_memory_limits(self) -> Dict[str, int]:
        """
        Calculate safe memory limits for different operations.
        
        Uses conservative allocation strategy:
        - 40% for model weights
        - 20% for context/activation memory
        - 10% for buffers and temporary data
        - 30% reserved for system
        """
        total_memory = self.device_info['total_memory']
        
        limits = {
            'model_memory': int(total_memory * 0.4),      # 40% for model
            'context_memory': int(total_memory * 0.2),    # 20% for context
            'buffer_memory': int(total_memory * 0.1),     # 10% for buffers
            'system_memory': int(total_memory * 0.3),     # 30% for system
        }
        
        return limits
    
    def get_profile(self) -> MemoryProfile:
        """Get structured memory profile"""
        
        info = self.device_info
        
        has_gpu = info.get('gpu_count', 0) > 0 or info.get('gpu_type') is not None
        gpu_memory = None
        
        if has_gpu and isinstance(info.get('gpu_memory'), list) and len(info['gpu_memory']) > 0:
            if isinstance(info['gpu_memory'][0], int):
                gpu_memory = info['gpu_memory'][0] / (1024 * 1024)  # Convert to MB
        
        return MemoryProfile(
            platform=info['platform'],
            architecture=info['architecture'],
            processor=info['processor'],
            total_memory_mb=info['total_memory'] / (1024 * 1024),
            available_memory_mb=info['available_memory'] / (1024 * 1024),
            cpu_count=info['cpu_count'],
            cpu_freq_mhz=info['cpu_freq']['current'] if info['cpu_freq'] else None,
            has_gpu=has_gpu,
            gpu_memory_mb=gpu_memory
        )
    
    def get_memory_recommendations(self, model_size_mb: float) -> Dict[str, Any]:
        """
        Get memory recommendations for a specific model size.
        
        Args:
            model_size_mb: Model size in megabytes
            
        Returns:
            Dictionary with recommendations
        """
        model_size_bytes = model_size_mb * 1024 * 1024
        
        recommendations = {
            'max_context_length': self._calculate_max_context_length(model_size_bytes),
            'optimal_batch_size': self._calculate_optimal_batch_size(model_size_bytes),
            'memory_safety_margin_mb': self.memory_limits['buffer_memory'] / (1024 * 1024),
            'recommended_quantization': self._recommend_quantization(model_size_bytes),
            'can_load_model': model_size_bytes <= self.memory_limits['model_memory'],
            'memory_pressure': self._calculate_memory_pressure(model_size_bytes)
        }
        
        return recommendations
    
    def _calculate_max_context_length(self, model_size: int) -> int:
        """
        Calculate maximum safe context length.
        
        Rule of thumb: 1 token ≈ 4 bytes in context memory
        """
        available_memory = self.memory_limits['context_memory']
        
        # Estimate: 1 token ≈ 4 bytes
        max_tokens = available_memory // 4
        
        # Apply 80% safety factor
        safe_tokens = int(max_tokens * 0.8)
        
        # Cap at 32K tokens (common maximum)
        return min(safe_tokens, 32768)
    
    def _calculate_optimal_batch_size(self, model_size: int) -> int:
        """Calculate optimal batch size for inference"""
        available_memory = self.memory_limits['model_memory']
        
        # Estimate memory per sample (model size + 50% overhead)
        memory_per_sample = model_size * 1.5
        
        max_batch_size = int(available_memory // memory_per_sample)
        
        # Clamp between 1 and 32
        return max(1, min(max_batch_size, 32))
    
    def _recommend_quantization(self, model_size: int) -> str:
        """Recommend quantization level based on model size and available memory"""
        
        available_model_memory = self.memory_limits['model_memory']
        
        if model_size > available_model_memory:
            return 'q4_k_m (4-bit) - REQUIRED for this device'
        elif model_size > available_model_memory * 0.7:
            return 'q8_0 (8-bit) - Recommended for optimal performance'
        else:
            return 'fp16 - No quantization needed'
    
    def _calculate_memory_pressure(self, model_size: int) -> str:
        """Calculate memory pressure level"""
        
        available_model_memory = self.memory_limits['model_memory']
        pressure_ratio = model_size / available_model_memory
        
        if pressure_ratio > 1.0:
            return 'CRITICAL - Model too large'
        elif pressure_ratio > 0.8:
            return 'HIGH - Consider smaller model or quantization'
        elif pressure_ratio > 0.5:
            return 'MODERATE - Monitor memory usage'
        else:
            return 'LOW - Comfortable headroom'
    
    def print_profile(self):
        """Print formatted device profile"""
        
        profile = self.get_profile()
        
        print("=" * 70)
        print("Device Memory Profile")
        print("=" * 70)
        print(f"Platform: {profile.platform}")
        print(f"Architecture: {profile.architecture}")
        print(f"Processor: {profile.processor}")
        print(f"Total Memory: {profile.total_memory_mb:.1f} MB")
        print(f"Available Memory: {profile.available_memory_mb:.1f} MB")
        print(f"CPU Cores: {profile.cpu_count}")
        
        if profile.cpu_freq_mhz:
            print(f"CPU Frequency: {profile.cpu_freq_mhz:.0f} MHz")
        
        if profile.has_gpu:
            print(f"GPU Available: Yes")
            if profile.gpu_memory_mb:
                print(f"GPU Memory: {profile.gpu_memory_mb:.1f} MB")
        else:
            print(f"GPU Available: No")
        
        print("\nMemory Allocation Strategy:")
        print(f"  Model Memory: {self.memory_limits['model_memory'] / (1024 * 1024):.1f} MB (40%)")
        print(f"  Context Memory: {self.memory_limits['context_memory'] / (1024 * 1024):.1f} MB (20%)")
        print(f"  Buffer Memory: {self.memory_limits['buffer_memory'] / (1024 * 1024):.1f} MB (10%)")
        print(f"  System Reserved: {self.memory_limits['system_memory'] / (1024 * 1024):.1f} MB (30%)")
        print("=" * 70)

def demonstrate_memory_profiling():
    """Demonstrate memory profiling capabilities"""
    
    print("Chapter 11: Edge Management - Memory Profiling Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print()
    
    # Create profiler
    profiler = DeviceMemoryProfiler()
    
    # Print device profile
    profiler.print_profile()
    
    # Test with different model sizes
    print("\nModel Size Analysis:")
    print("=" * 70)
    
    test_models = [
        ("TinyLlama-1.1B (Q4)", 700),      # 700 MB
        ("Qwen3-4B (Q4)", 2500),           # 2.5 GB
        ("Llama-7B (Q4)", 4000),           # 4 GB
        ("Mistral-7B (Q8)", 7000),         # 7 GB
    ]
    
    for model_name, model_size_mb in test_models:
        print(f"\n{model_name} ({model_size_mb} MB):")
        print("-" * 70)
        
        recommendations = profiler.get_memory_recommendations(model_size_mb)
        
        can_load = "✅ YES" if recommendations['can_load_model'] else "❌ NO"
        print(f"  Can Load: {can_load}")
        print(f"  Memory Pressure: {recommendations['memory_pressure']}")
        print(f"  Recommended Quantization: {recommendations['recommended_quantization']}")
        print(f"  Max Context Length: {recommendations['max_context_length']:,} tokens")
        print(f"  Optimal Batch Size: {recommendations['optimal_batch_size']}")
        print(f"  Safety Margin: {recommendations['memory_safety_margin_mb']:.1f} MB")
    
    print("\n" + "=" * 70)
    print("💡 Key Takeaways:")
    print("  • Always profile your target device before deployment")
    print("  • Use conservative memory allocation (40% for model)")
    print("  • Monitor memory pressure during inference")
    print("  • Adjust quantization based on device constraints")
        print("=" * 70)

class DeviceMemoryManager:
    """
    Advanced memory manager with watermark system and UMA support.
    
    Implements the three non-negotiable boundaries:
    1. Model Weights Size
    2. Safety Margin (5-10% buffer)
    3. KV Cache Budget
    """
    
    def __init__(self, profiler: DeviceMemoryProfiler, config: WatermarkConfig = None):
        """Initialize device memory manager"""
        self.profiler = profiler
        self.config = config or WatermarkConfig()
        self.memory_callbacks: List[Callable] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.current_watermark = WatermarkLevel.LOW
        self.degradation_level = 0
        self.max_degradation_level = 3
        
        # Memory allocation boundaries
        self.model_memory_limit = 0
        self.safety_margin = 0
        self.kv_cache_budget = 0
        self._calculate_memory_boundaries()
    
    def _calculate_memory_boundaries(self):
        """Calculate the three non-negotiable memory boundaries"""
        total_memory = self.profiler.device_info['total_memory']
        
        # Safety margin: 5-10% of total memory
        self.safety_margin = int(total_memory * self.config.safety_margin)
        
        # Model memory: 40% of available memory (after safety margin)
        available_after_safety = total_memory - self.safety_margin
        self.model_memory_limit = int(available_after_safety * 0.4)
        
        # KV Cache budget: remaining memory for dynamic operations
        self.kv_cache_budget = available_after_safety - self.model_memory_limit
        
        print(f"Memory Boundaries Calculated:")
        print(f"  Total Memory: {total_memory / (1024**3):.1f} GB")
        print(f"  Safety Margin: {self.safety_margin / (1024**3):.1f} GB ({self.config.safety_margin:.0%})")
        print(f"  Model Limit: {self.model_memory_limit / (1024**3):.1f} GB")
        print(f"  KV Cache Budget: {self.kv_cache_budget / (1024**3):.1f} GB")
    
    def start_monitoring(self):
        """Start memory watermark monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        print("🔍 Memory watermark monitoring started")
    
    def stop_monitoring(self):
        """Stop memory watermark monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️ Memory watermark monitoring stopped")
    
    def _monitor_memory(self):
        """Monitor memory usage and trigger watermarks"""
        while self.monitoring:
            try:
                memory_usage = self._get_memory_usage()
                watermark = self._get_watermark_level(memory_usage)
                
                if watermark != self.current_watermark:
                    self._handle_watermark_change(watermark, memory_usage)
                    self.current_watermark = watermark
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                print(f"❌ Memory monitoring error: {e}")
                time.sleep(self.config.check_interval)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as fraction of total"""
        memory = psutil.virtual_memory()
        return memory.used / memory.total
    
    def _get_watermark_level(self, usage: float) -> WatermarkLevel:
        """Determine watermark level based on memory usage"""
        if usage >= self.config.high_threshold:
            return WatermarkLevel.HIGH
        elif usage >= self.config.medium_threshold:
            return WatermarkLevel.MEDIUM
        elif usage >= self.config.low_threshold:
            return WatermarkLevel.LOW
        else:
            return WatermarkLevel.LOW
    
    def _handle_watermark_change(self, watermark: WatermarkLevel, usage: float):
        """Handle watermark level changes"""
        print(f"🚨 Memory Watermark: {watermark.value.upper()} ({usage:.1%} usage)")
        
        if watermark == WatermarkLevel.LOW:
            self._trigger_cleanup()
        elif watermark == WatermarkLevel.MEDIUM:
            self._trigger_level_1_degradation()
        elif watermark == WatermarkLevel.HIGH:
            self._trigger_level_2_degradation()
        
        # Notify callbacks
        for callback in self.memory_callbacks:
            try:
                callback(watermark, usage)
            except Exception as e:
                print(f"❌ Memory callback error: {e}")
    
    def _trigger_cleanup(self):
        """Trigger minor cleanup and garbage collection"""
        print("🧹 Triggering cleanup (Level 0)")
        gc.collect()
        
        # Clear PyTorch cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _trigger_level_1_degradation(self):
        """Trigger Level 1 degradation: reduce batch size, clear caches"""
        print("⚠️ Triggering Level 1 Degradation")
        self.degradation_level = 1
        
        # Reduce batch size
        # Clear temporary caches
        # Sacrifice throughput for stability
        self._trigger_cleanup()
    
    def _trigger_level_2_degradation(self):
        """Trigger Level 2 degradation: aggressive context truncation"""
        print("🚨 Triggering Level 2 Degradation")
        self.degradation_level = 2
        
        # Aggressive context truncation
        # Sacrifice history for system survival
        self._trigger_cleanup()
    
    def add_memory_callback(self, callback: Callable):
        """Add callback for memory events"""
        self.memory_callbacks.append(callback)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        memory = psutil.virtual_memory()
        
        return {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'used_memory_gb': memory.used / (1024**3),
            'usage_percentage': memory.percent,
            'current_watermark': self.current_watermark.value,
            'degradation_level': self.degradation_level,
            'model_memory_limit_gb': self.model_memory_limit / (1024**3),
            'kv_cache_budget_gb': self.kv_cache_budget / (1024**3),
            'safety_margin_gb': self.safety_margin / (1024**3)
        }
    
    def can_load_model(self, model_size_bytes: int) -> bool:
        """Check if model can be loaded within memory limits"""
        return model_size_bytes <= self.model_memory_limit
    
    def get_optimal_context_size(self, model_size_bytes: int) -> int:
        """Calculate optimal context size based on available KV cache budget"""
        if not self.can_load_model(model_size_bytes):
            return 0
        
        # Estimate KV cache cost per token (rough: 128 bytes per token)
        bytes_per_token = 128
        available_for_kv = self.kv_cache_budget - (model_size_bytes - self.model_memory_limit)
        
        if available_for_kv <= 0:
            return 0
        
        max_tokens = int(available_for_kv / bytes_per_token)
        
        # Apply safety factor
        safe_tokens = int(max_tokens * 0.8)
        
        # Round down to common context sizes
        context_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        return max([size for size in context_sizes if size <= safe_tokens], default=512)

def demonstrate_advanced_memory_management():
    """Demonstrate advanced memory management with watermarks"""
    
    print("=" * 70)
    print("Chapter 11: Advanced Memory Management Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Create profiler and memory manager
    profiler = DeviceMemoryProfiler()
    memory_manager = DeviceMemoryManager(profiler)
    
    # Print memory boundaries
    print("\n📊 Memory Boundaries:")
    print("-" * 30)
    status = memory_manager.get_memory_status()
    print(f"Total Memory: {status['total_memory_gb']:.1f} GB")
    print(f"Safety Margin: {status['safety_margin_gb']:.1f} GB")
    print(f"Model Limit: {status['model_memory_limit_gb']:.1f} GB")
    print(f"KV Cache Budget: {status['kv_cache_budget_gb']:.1f} GB")
    
    # Test model loading scenarios
    print("\n🧪 Model Loading Scenarios:")
    print("-" * 30)
    
    test_models = [
        ("TinyLlama-1.1B (Q4)", 700 * 1024 * 1024),      # 700 MB
        ("Qwen3-4B (Q4)", 2500 * 1024 * 1024),           # 2.5 GB
        ("Llama-7B (Q4)", 4000 * 1024 * 1024),           # 4 GB
        ("Mistral-7B (Q8)", 7000 * 1024 * 1024),         # 7 GB
    ]
    
    for model_name, model_size in test_models:
        can_load = memory_manager.can_load_model(model_size)
        context_size = memory_manager.get_optimal_context_size(model_size)
        
        status_icon = "✅" if can_load else "❌"
        print(f"{status_icon} {model_name}:")
        print(f"  Can Load: {can_load}")
        if can_load:
            print(f"  Optimal Context: {context_size:,} tokens")
        else:
            print(f"  Reason: Model too large for device")
    
    # Demonstrate watermark monitoring
    print("\n🔍 Watermark Monitoring Demo:")
    print("-" * 30)
    print("Starting memory monitoring (simulated)...")
    
    # Add callback to demonstrate watermark changes
    def watermark_callback(watermark, usage):
        print(f"  📊 Watermark changed to {watermark.value.upper()} at {usage:.1%} usage")
    
    memory_manager.add_memory_callback(watermark_callback)
    
    # Start monitoring
    memory_manager.start_monitoring()
    
    # Simulate memory pressure (in real scenario, this would be actual memory usage)
    print("  Simulating memory pressure scenarios...")
    time.sleep(2)
    
    # Stop monitoring
    memory_manager.stop_monitoring()
    
    print("\n💡 Key Takeaways:")
    print("  • Watermark system prevents OOM crashes")
    print("  • Three-tiered degradation maintains stability")
    print("  • UMA support handles unified memory architectures")
    print("  • Real-time monitoring enables proactive management")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_memory_profiling()
    print("\n" + "="*70 + "\n")
    demonstrate_advanced_memory_management()

