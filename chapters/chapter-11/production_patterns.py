#!/usr/bin/env python3
"""
Chapter 11: Edge Management - Production Architecture Patterns

This module demonstrates production-ready error handling, graceful degradation,
and state persistence systems for sovereign edge AI agents.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import logging
import traceback
import json
import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import os

class ErrorSeverity(Enum):
    """Error severity levels for production systems"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProductionErrorHandler:
    """
    Production error handler with graceful degradation.
    
    Implements systematic error response using ErrorSeverity enumeration
    to prevent system collapse from single failures.
    """
    
    def __init__(self, log_level=logging.INFO):
        """Initialize production error handler"""
        self.logger = self._setup_logger(log_level)
        self.error_counts = {}
        self.degradation_strategies = {
            ErrorSeverity.LOW: self._handle_low_severity,
            ErrorSeverity.MEDIUM: self._handle_medium_severity,
            ErrorSeverity.HIGH: self._handle_high_severity,
            ErrorSeverity.CRITICAL: self._handle_critical_severity
        }
        self.retry_counts = {}
        self.max_retries = 3
    
    def _setup_logger(self, log_level):
        """Setup production logger"""
        
        logger = logging.getLogger('on_device_ai')
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = logging.FileHandler('ai_engine.log')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def handle_error(self, error: Exception, context: Dict[str, Any], severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Handle error with appropriate strategy"""
        
        # Log error
        self.logger.error(f"Error occurred: {str(error)}", extra={
            'context': context,
            'severity': severity.value,
            'traceback': traceback.format_exc()
        })
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Apply degradation strategy
        strategy = self.degradation_strategies.get(severity)
        if strategy:
            strategy(error, context)
    
    def _handle_low_severity(self, error: Exception, context: Dict[str, Any]):
        """Handle low severity errors"""
        
        # Log and continue
        self.logger.info(f"Low severity error handled: {str(error)}")
    
    def _handle_medium_severity(self, error: Exception, context: Dict[str, Any]):
        """Handle medium severity errors"""
        
        # Log and potentially retry
        self.logger.warning(f"Medium severity error: {str(error)}")
        
        # Check if we should retry
        if self._should_retry(error, context):
            self._schedule_retry(context)
    
    def _handle_high_severity(self, error: Exception, context: Dict[str, Any]):
        """Handle high severity errors"""
        
        # Log and degrade performance
        self.logger.error(f"High severity error: {str(error)}")
        
        # Apply performance degradation
        self._degrade_performance()
    
    def _handle_critical_severity(self, error: Exception, context: Dict[str, Any]):
        """Handle critical severity errors"""
        
        # Log and shutdown gracefully
        self.logger.critical(f"Critical error: {str(error)}")
        
        # Initiate graceful shutdown
        self._initiate_graceful_shutdown()
    
    def _should_retry(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Determine if operation should be retried"""
        
        # Don't retry if we've already retried too many times
        retry_count = context.get('retry_count', 0)
        if retry_count >= self.max_retries:
            return False
        
        # Don't retry certain types of errors
        non_retryable_errors = [ValueError, TypeError, AttributeError]
        if type(error) in non_retryable_errors:
            return False
        
        return True
    
    def _schedule_retry(self, context: Dict[str, Any]):
        """Schedule retry of failed operation"""
        
        retry_count = context.get('retry_count', 0) + 1
        context['retry_count'] = retry_count
        
        # Exponential backoff
        delay = 2 ** retry_count
        
        self.logger.info(f"Scheduling retry {retry_count} in {delay} seconds")
        
        # Schedule retry (simplified - would use actual scheduler)
        timer = threading.Timer(delay, self._execute_retry, args=[context])
        timer.start()
    
    def _execute_retry(self, context: Dict[str, Any]):
        """Execute retry of failed operation"""
        
        try:
            # Re-execute the operation
            operation = context.get('operation')
            if operation:
                operation()
        except Exception as e:
            # Handle retry failure
            self.handle_error(e, context, ErrorSeverity.HIGH)
    
    def _degrade_performance(self):
        """Degrade performance to handle errors"""
        
        self.logger.warning("Degrading performance due to errors")
        
        # Reduce batch size
        # Reduce context length
        # Enable more aggressive quantization
        # Clear caches
        
        pass
    
    def _initiate_graceful_shutdown(self):
        """Initiate graceful shutdown"""
        
        self.logger.critical("Initiating graceful shutdown")
        
        # Stop accepting new requests
        # Finish processing current requests
        # Save state
        # Clean up resources
        
        pass
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        
        return {
            'error_counts': self.error_counts,
            'total_errors': sum(self.error_counts.values()),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }

class GracefulDegradationManager:
    """
    Graceful degradation manager for edge AI systems.
    
    Manages system degradation levels to maintain service availability
    under resource constraints and error conditions.
    """
    
    def __init__(self, error_handler: ProductionErrorHandler):
        """Initialize graceful degradation manager"""
        self.error_handler = error_handler
        self.degradation_level = 0
        self.max_degradation_level = 3
        self.degradation_callbacks = []
        
    def increase_degradation(self):
        """Increase degradation level"""
        
        if self.degradation_level < self.max_degradation_level:
            self.degradation_level += 1
            self._apply_degradation()
            
            # Notify callbacks
            for callback in self.degradation_callbacks:
                try:
                    callback(self.degradation_level, 'increased')
                except Exception as e:
                    self.error_handler.logger.error(f"Degradation callback error: {e}")
    
    def decrease_degradation(self):
        """Decrease degradation level"""
        
        if self.degradation_level > 0:
            self.degradation_level -= 1
            self._apply_degradation()
            
            # Notify callbacks
            for callback in self.degradation_callbacks:
                try:
                    callback(self.degradation_level, 'decreased')
                except Exception as e:
                    self.error_handler.logger.error(f"Degradation callback error: {e}")
    
    def _apply_degradation(self):
        """Apply current degradation level"""
        
        if self.degradation_level == 0:
            # No degradation
            self._restore_full_performance()
        elif self.degradation_level == 1:
            # Light degradation
            self._reduce_batch_size()
        elif self.degradation_level == 2:
            # Medium degradation
            self._reduce_context_length()
        elif self.degradation_level == 3:
            # Heavy degradation
            self._enable_quantization()
    
    def _restore_full_performance(self):
        """Restore full performance"""
        
        self.error_handler.logger.info("Restoring full performance")
        
        # Restore original settings
        pass
    
    def _reduce_batch_size(self):
        """Reduce batch size"""
        
        self.error_handler.logger.info("Reducing batch size")
        
        # Reduce batch size by 50%
        pass
    
    def _reduce_context_length(self):
        """Reduce context length"""
        
        self.error_handler.logger.info("Reducing context length")
        
        # Reduce context length by 50%
        pass
    
    def _enable_quantization(self):
        """Enable quantization"""
        
        self.error_handler.logger.info("Enabling quantization")
        
        # Enable aggressive quantization
        pass
    
    def add_degradation_callback(self, callback: Callable):
        """Add callback for degradation events"""
        self.degradation_callbacks.append(callback)
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'degradation_level': self.degradation_level,
            'max_degradation_level': self.max_degradation_level,
            'status': 'degraded' if self.degradation_level > 0 else 'normal'
        }

class StatePersistenceManager:
    """
    State persistence manager for sovereign agents.
    
    Enables agents to survive reboots and maintain autonomy
    through persistent state management.
    """
    
    def __init__(self, storage_path: str = "agent_state.json"):
        """Initialize state persistence manager"""
        self.storage_path = storage_path
        self.state = {}
        self.lock = threading.Lock()
        self.auto_save_interval = 30  # seconds
        self.auto_save_timer = None
        self._start_auto_save()
    
    def _start_auto_save(self):
        """Start automatic state saving"""
        if self.auto_save_timer:
            self.auto_save_timer.cancel()
        
        self.auto_save_timer = threading.Timer(self.auto_save_interval, self._auto_save)
        self.auto_save_timer.daemon = True
        self.auto_save_timer.start()
    
    def _auto_save(self):
        """Automatically save state"""
        try:
            self.save_state()
        except Exception as e:
            print(f"Auto-save error: {e}")
        finally:
            self._start_auto_save()  # Schedule next auto-save
    
    def set_state(self, key: str, value: Any):
        """Set state value"""
        with self.lock:
            self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        with self.lock:
            return self.state.get(key, default)
    
    def save_state(self):
        """Save state to persistent storage"""
        with self.lock:
            try:
                with open(self.storage_path, 'w') as f:
                    json.dump(self.state, f, indent=2, default=str)
                print(f"✅ State saved to {self.storage_path}")
            except Exception as e:
                print(f"❌ Failed to save state: {e}")
    
    def load_state(self):
        """Load state from persistent storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.state = json.load(f)
                print(f"✅ State loaded from {self.storage_path}")
                return True
            else:
                print(f"ℹ️ No existing state file at {self.storage_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to load state: {e}")
            return False
    
    def clear_state(self):
        """Clear all state"""
        with self.lock:
            self.state.clear()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary"""
        with self.lock:
            return {
                'keys': list(self.state.keys()),
                'size': len(self.state),
                'storage_path': self.storage_path,
                'auto_save_interval': self.auto_save_interval
            }

def demonstrate_production_patterns():
    """Demonstrate production architecture patterns"""
    
    print("=" * 70)
    print("Chapter 11: Production Architecture Patterns Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Demo 1: Error Handling
    print("\n🚨 Demo 1: Error Handling & Graceful Degradation")
    print("-" * 50)
    
    error_handler = ProductionErrorHandler()
    degradation_manager = GracefulDegradationManager(error_handler)
    
    # Simulate different error scenarios
    print("Simulating error scenarios...")
    
    # Low severity error
    try:
        raise ValueError("Minor configuration issue")
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'config_check'}, ErrorSeverity.LOW)
    
    # Medium severity error
    try:
        raise ConnectionError("Network timeout")
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'network_request'}, ErrorSeverity.MEDIUM)
    
    # High severity error
    try:
        raise MemoryError("Out of memory")
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'model_inference'}, ErrorSeverity.HIGH)
        degradation_manager.increase_degradation()
    
    # Show error statistics
    stats = error_handler.get_error_stats()
    print(f"Error Statistics: {stats}")
    
    # Demo 2: State Persistence
    print("\n💾 Demo 2: State Persistence")
    print("-" * 50)
    
    state_manager = StatePersistenceManager("demo_agent_state.json")
    
    # Set some state
    state_manager.set_state("current_goal", "Help user with Python programming")
    state_manager.set_state("conversation_history", ["Hello", "How can I help?"])
    state_manager.set_state("last_activity", time.time())
    state_manager.set_state("degradation_level", degradation_manager.degradation_level)
    
    # Save state
    state_manager.save_state()
    
    # Load state in new instance
    new_state_manager = StatePersistenceManager("demo_agent_state.json")
    new_state_manager.load_state()
    
    # Show loaded state
    print("Loaded state:")
    for key, value in new_state_manager.state.items():
        print(f"  {key}: {value}")
    
    # Demo 3: Degradation Management
    print("\n⚡ Demo 3: Degradation Management")
    print("-" * 50)
    
    # Add degradation callback
    def degradation_callback(level, action):
        print(f"  📊 Degradation {action} to level {level}")
    
    degradation_manager.add_degradation_callback(degradation_callback)
    
    # Simulate degradation scenarios
    print("Simulating resource pressure...")
    degradation_manager.increase_degradation()
    degradation_manager.increase_degradation()
    
    status = degradation_manager.get_degradation_status()
    print(f"Degradation Status: {status}")
    
    # Restore performance
    degradation_manager.decrease_degradation()
    degradation_manager.decrease_degradation()
    
    # Clean up
    if os.path.exists("demo_agent_state.json"):
        os.remove("demo_agent_state.json")
        print("🧹 Cleaned up demo state file")
    
    print("\n💡 Key Takeaways:")
    print("  • Systematic error handling prevents system collapse")
    print("  • Graceful degradation maintains service availability")
    print("  • State persistence enables true autonomy")
    print("  • Production patterns ensure reliability")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_production_patterns()
