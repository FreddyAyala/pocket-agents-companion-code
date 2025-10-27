#!/usr/bin/env python3
"""
Chapter 11: Edge Management - Context Window Optimization

This module demonstrates context window management strategies for edge AI,
including dynamic sizing, compression, and the 500-token buffer strategy.
Implements the three-tiered context compression system with Summary Chain
and Semantic Memory Injection.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class ContextConfig:
    """Configuration for context window management"""
    max_context_tokens: int
    buffer_tokens: int = 500  # The 500-token buffer strategy
    system_prompt_tokens: int = 0
    compression_threshold: float = 0.9  # Compress at 90% capacity

class ContextWindowManager:
    """
    Context window manager for edge AI deployment.
    
    Implements dynamic context sizing, compression, and auto-clearing
    strategies to maintain optimal memory usage.
    """
    
    def __init__(self, config: ContextConfig):
        """Initialize context window manager"""
        self.config = config
        self.conversation_history: deque = deque()
        self.token_count = 0
        self.total_messages = 0
        self.compressions_performed = 0
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens in context window"""
        reserved = self.config.system_prompt_tokens + self.config.buffer_tokens
        return self.config.max_context_tokens - self.token_count - reserved
    
    @property
    def utilization(self) -> float:
        """Get context window utilization (0.0 to 1.0)"""
        return self.token_count / self.config.max_context_tokens
    
    def add_message(self, role: str, content: str, token_count: int):
        """
        Add message to conversation history.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            token_count: Estimated token count
        """
        message = {
            'role': role,
            'content': content,
            'tokens': token_count,
            'timestamp': time.time()
        }
        
        self.conversation_history.append(message)
        self.token_count += token_count
        self.total_messages += 1
        
        # Check if compression needed
        if self.utilization >= self.config.compression_threshold:
            self._compress_context()
    
    def _compress_context(self):
        """
        Compress context using the 500-token buffer strategy.
        
        Keeps:
        - System prompt (always)
        - Last N messages that fit within max_context - buffer
        - Ensures at least one conversation turn is preserved
        """
        print(f"⚠️ Context window at {self.utilization:.1%} capacity. Compressing...")
        
        # Calculate target token count (max - buffer)
        target_tokens = self.config.max_context_tokens - self.config.buffer_tokens
        
        # Keep most recent messages that fit
        compressed_history = deque()
        compressed_tokens = self.config.system_prompt_tokens
        
        # Iterate from most recent to oldest
        for message in reversed(self.conversation_history):
            if compressed_tokens + message['tokens'] <= target_tokens:
                compressed_history.appendleft(message)
                compressed_tokens += message['tokens']
            else:
                break
        
        # Ensure at least one user-assistant pair is kept
        if len(compressed_history) < 2:
            # Keep last 2 messages regardless
            compressed_history = deque(list(self.conversation_history)[-2:])
            compressed_tokens = sum(msg['tokens'] for msg in compressed_history)
            compressed_tokens += self.config.system_prompt_tokens
        
        # Update history
        messages_removed = len(self.conversation_history) - len(compressed_history)
        tokens_freed = self.token_count - (compressed_tokens - self.config.system_prompt_tokens)
        
        self.conversation_history = compressed_history
        self.token_count = compressed_tokens - self.config.system_prompt_tokens
        self.compressions_performed += 1
        
        print(f"✅ Compressed: Removed {messages_removed} messages, freed {tokens_freed} tokens")
        print(f"   New utilization: {self.utilization:.1%}")
    
    def get_context_for_inference(self) -> List[Dict[str, str]]:
        """Get formatted context for model inference"""
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in self.conversation_history
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context window statistics"""
        return {
            'max_context_tokens': self.config.max_context_tokens,
            'current_tokens': self.token_count,
            'available_tokens': self.available_tokens,
            'utilization': self.utilization,
            'buffer_tokens': self.config.buffer_tokens,
            'total_messages': self.total_messages,
            'active_messages': len(self.conversation_history),
            'compressions_performed': self.compressions_performed
        }
    
    def print_status(self):
        """Print formatted status"""
        stats = self.get_statistics()
        
        print(f"\nContext Window Status:")
        print(f"  Utilization: {stats['utilization']:.1%} ({stats['current_tokens']}/{stats['max_context_tokens']} tokens)")
        print(f"  Available: {stats['available_tokens']:,} tokens")
        print(f"  Buffer: {stats['buffer_tokens']} tokens")
        print(f"  Messages: {stats['active_messages']} active (of {stats['total_messages']} total)")
        print(f"  Compressions: {stats['compressions_performed']}")

class DynamicContextSizer:
    """
    Dynamic context sizing based on device capabilities.
    
    Adjusts context window size based on available memory and model size.
    """
    
    @staticmethod
    def calculate_optimal_context_size(
        available_memory_mb: float,
        model_size_mb: float,
        desired_size: int = 4096
    ) -> Tuple[int, str]:
        """
        Calculate optimal context size for device.
        
        Args:
            available_memory_mb: Available memory in MB
            model_size_mb: Model size in MB
            desired_size: Desired context size (default 4096)
            
        Returns:
            Tuple of (optimal_size, reasoning)
        """
        # Memory required per token (rough estimate)
        # KV cache: ~4 bytes per token per layer for fp16
        # For typical 32-layer model: ~128 bytes per token
        bytes_per_token = 128
        
        # Calculate available memory after model loading
        available_after_model = (available_memory_mb - model_size_mb) * 1024 * 1024
        
        # Calculate max tokens that fit
        max_tokens = int(available_after_model / bytes_per_token)
        
        # Apply safety factor (80%)
        safe_tokens = int(max_tokens * 0.8)
        
        # Round down to nearest power of 2 (common context sizes)
        context_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        optimal_size = max([size for size in context_sizes if size <= safe_tokens], default=512)
        
        # Clamp to desired size if possible
        optimal_size = min(optimal_size, desired_size)
        
        # Determine reasoning
        if optimal_size >= desired_size:
            reasoning = f"Comfortable: {optimal_size} tokens supported"
        elif optimal_size >= 2048:
            reasoning = f"Adequate: {optimal_size} tokens (consider quantization for larger context)"
        else:
            reasoning = f"Constrained: {optimal_size} tokens (aggressive quantization recommended)"
        
        return optimal_size, reasoning

class ThreeTieredContextSystem:
    """
    Three-tiered dynamic context system for edge AI.
    
    Implements:
    - Tier 1: Raw Buffer (500-token short-term memory)
    - Tier 2: Compressed Summary Chain (long-term sequential memory)
    - Tier 3: Semantic Memory Injection (non-sequential recall)
    """
    
    def __init__(self, config: ContextConfig):
        """Initialize three-tiered context system"""
        self.config = config
        self.raw_buffer: deque = deque(maxlen=500)  # Tier 1: Raw Buffer
        self.summary_chain: List[str] = []          # Tier 2: Summary Chain
        self.semantic_index: Dict[str, Any] = {}    # Tier 3: Semantic Index
        self.compression_count = 0
        
    def add_message(self, role: str, content: str, token_count: int):
        """Add message to the three-tiered system"""
        message = {
            'role': role,
            'content': content,
            'tokens': token_count,
            'timestamp': time.time(),
            'id': hashlib.md5(content.encode()).hexdigest()[:8]
        }
        
        # Add to raw buffer (Tier 1)
        self.raw_buffer.append(message)
        
        # Check if compression needed
        if self._needs_compression():
            self._compress_context()
    
    def _needs_compression(self) -> bool:
        """Check if context compression is needed"""
        total_tokens = sum(msg['tokens'] for msg in self.raw_buffer)
        return total_tokens >= self.config.max_context_tokens * 0.9  # 90% threshold
    
    def _compress_context(self):
        """Compress context using three-tiered approach"""
        print("🔄 Triggering three-tiered context compression...")
        
        # Get messages to compress (exclude recent 500 tokens)
        messages_to_compress = list(self.raw_buffer)[:-500] if len(self.raw_buffer) > 500 else []
        
        if not messages_to_compress:
            return
        
        # Tier 2: Create Summary Chain
        summary_chain = self._create_summary_chain(messages_to_compress)
        self.summary_chain.extend(summary_chain)
        
        # Tier 3: Update Semantic Index
        self._update_semantic_index(messages_to_compress)
        
        # Remove compressed messages from raw buffer
        for _ in range(len(messages_to_compress)):
            if self.raw_buffer:
                self.raw_buffer.popleft()
        
        self.compression_count += 1
        print(f"✅ Compression complete: {len(summary_chain)} summary tokens, {len(messages_to_compress)} messages processed")
    
    def _create_summary_chain(self, messages: List[Dict]) -> List[str]:
        """Create summary chain from messages (Tier 2)"""
        # Simulate summarizer SLM output
        conversation_text = " ".join([msg['content'] for msg in messages])
        
        # Simple extractive summarization (in production, use actual SLM)
        sentences = conversation_text.split('. ')
        key_sentences = sentences[:3]  # Take first 3 sentences as summary
        
        summary_tokens = []
        for sentence in key_sentences:
            if sentence.strip():
                summary_tokens.append(f"[SUMMARY] {sentence.strip()}")
        
        return summary_tokens
    
    def _update_semantic_index(self, messages: List[Dict]):
        """Update semantic index with key entities (Tier 3)"""
        for message in messages:
            content = message['content']
            
            # Extract entities (in production, use NER model)
            entities = self._extract_entities(content)
            
            # Create embeddings (simplified)
            for entity in entities:
                embedding = self._create_embedding(entity)
                self.semantic_index[entity] = {
                    'embedding': embedding,
                    'timestamp': message['timestamp'],
                    'context': content[:100] + "..." if len(content) > 100 else content
                }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified)"""
        # Simple keyword extraction (in production, use NER)
        keywords = []
        words = text.lower().split()
        
        # Look for important words
        important_words = ['python', 'code', 'function', 'variable', 'class', 'method', 
                         'error', 'bug', 'fix', 'problem', 'solution', 'help', 'learn']
        
        for word in words:
            if word in important_words and word not in keywords:
                keywords.append(word)
        
        return keywords[:5]  # Limit to 5 entities per message
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text (simplified)"""
        # Simple hash-based embedding (in production, use actual embedding model)
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 128-dimensional vector
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8)
        embedding = np.tile(embedding, 16)[:128]  # Repeat to get 128 dimensions
        embedding = embedding.astype(np.float32) / 255.0  # Normalize
        
        return embedding
    
    def get_semantic_matches(self, query: str, top_k: int = 3) -> List[str]:
        """Get semantic matches for query (Tier 3)"""
        if not self.semantic_index:
            return []
        
        query_embedding = self._create_embedding(query)
        similarities = []
        
        for entity, data in self.semantic_index.items():
            embedding = data['embedding']
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((entity, similarity, data))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:top_k]]
    
    def get_final_prompt(self, user_query: str) -> str:
        """Get final prompt with three-tiered context injection"""
        
        # A: System Instructions and Persona
        system_prompt = "You are a helpful AI assistant with access to conversation history."
        
        # B: Tier 2: Compressed Summary Chain
        summary_section = ""
        if self.summary_chain:
            summary_section = "[CONVERSATION SUMMARY]\n" + "\n".join(self.summary_chain[-5:])  # Last 5 summaries
        
        # C: Tier 3: Semantic Memory Injection
        semantic_section = ""
        semantic_matches = self.get_semantic_matches(user_query)
        if semantic_matches:
            semantic_section = f"[SEMANTIC MEMORY INJECTION: {', '.join(semantic_matches)}]"
        
        # D: Tier 1: Raw Buffer
        raw_buffer_section = ""
        if self.raw_buffer:
            recent_messages = list(self.raw_buffer)[-10:]  # Last 10 messages
            raw_buffer_section = "[RECENT CONVERSATION]\n"
            for msg in recent_messages:
                raw_buffer_section += f"{msg['role']}: {msg['content']}\n"
        
        # E: User Query
        user_section = f"[USER QUERY]\n{user_query}"
        
        # Combine all sections
        final_prompt = f"{system_prompt}\n\n{summary_section}\n\n{semantic_section}\n\n{raw_buffer_section}\n\n{user_section}"
        
        return final_prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'raw_buffer_messages': len(self.raw_buffer),
            'raw_buffer_tokens': sum(msg['tokens'] for msg in self.raw_buffer),
            'summary_chain_length': len(self.summary_chain),
            'semantic_index_size': len(self.semantic_index),
            'compressions_performed': self.compression_count,
            'total_capacity_utilization': sum(msg['tokens'] for msg in self.raw_buffer) / self.config.max_context_tokens
        }

def demonstrate_context_optimization():
    """Demonstrate context window optimization"""
    
    print("=" * 70)
    print("Chapter 11: Context Window Optimization Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Demo 1: Dynamic Context Sizing
    print("\n📊 Demo 1: Dynamic Context Sizing")
    print("-" * 70)
    
    test_configs = [
        ("iPhone 13", 4000, 700),      # 4GB RAM, 700MB model
        ("iPad Pro", 8000, 2500),       # 8GB RAM, 2.5GB model
        ("MacBook M3", 16000, 4000),    # 16GB RAM, 4GB model
    ]
    
    for device_name, available_memory, model_size in test_configs:
        optimal_size, reasoning = DynamicContextSizer.calculate_optimal_context_size(
            available_memory, model_size
        )
        print(f"\n{device_name}:")
        print(f"  Available Memory: {available_memory} MB")
        print(f"  Model Size: {model_size} MB")
        print(f"  Optimal Context: {optimal_size} tokens")
        print(f"  Reasoning: {reasoning}")
    
    # Demo 2: 500-Token Buffer Strategy
    print("\n\n📊 Demo 2: 500-Token Buffer Strategy")
    print("-" * 70)
    
    # Create context manager with 2048 token limit
    config = ContextConfig(
        max_context_tokens=2048,
        buffer_tokens=500,
        system_prompt_tokens=100,
        compression_threshold=0.9
    )
    
    manager = ContextWindowManager(config)
    
    print(f"Configuration:")
    print(f"  Max Context: {config.max_context_tokens} tokens")
    print(f"  Buffer: {config.buffer_tokens} tokens")
    print(f"  System Prompt: {config.system_prompt_tokens} tokens")
    print(f"  Compression Threshold: {config.compression_threshold:.0%}")
    
    # Simulate conversation
    print(f"\nSimulating conversation...")
    
    conversations = [
        ("user", "Hello! Can you help me with Python programming?", 15),
        ("assistant", "Of course! I'd be happy to help you with Python. What specific topic would you like to learn about?", 25),
        ("user", "I want to understand list comprehensions and how they work.", 15),
        ("assistant", "List comprehensions are a concise way to create lists in Python. They follow the pattern: [expression for item in iterable if condition]. For example: squares = [x**2 for x in range(10)]", 45),
        ("user", "Can you show me more examples with conditions?", 12),
        ("assistant", "Sure! Here are examples: even_numbers = [x for x in range(20) if x % 2 == 0], filtered_words = [word for word in words if len(word) > 3]", 40),
        ("user", "How do nested list comprehensions work?", 10),
        ("assistant", "Nested list comprehensions let you work with nested structures. Example: matrix = [[i*j for j in range(3)] for i in range(3)] creates a multiplication table.", 35),
        ("user", "What about dictionary comprehensions?", 8),
        ("assistant", "Dictionary comprehensions are similar: {key: value for item in iterable}. Example: squares_dict = {x: x**2 for x in range(5)} creates {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}.", 45),
        ("user", "Can you show me performance comparisons?", 10),
        ("assistant", "List comprehensions are generally faster than loops. Benchmarks show 1.5-2x speed improvement for simple operations. Use timeit module to measure: timeit.timeit('[x**2 for x in range(100)]', number=10000)", 45),
        ("user", "When should I use regular loops instead?", 10),
        ("assistant", "Use regular loops when: logic is complex, you need multiple statements, readability matters more than speed, or you're doing side effects. List comprehensions are best for simple transformations.", 40),
    ]
    
    for i, (role, content, tokens) in enumerate(conversations, 1):
        print(f"\nTurn {i}: {role} ({tokens} tokens)")
        manager.add_message(role, content, tokens)
        manager.print_status()
    
    # Show final statistics
    print("\n\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    stats = manager.get_statistics()
    
    print(f"Total Messages Processed: {stats['total_messages']}")
    print(f"Active Messages: {stats['active_messages']}")
    print(f"Messages Removed: {stats['total_messages'] - stats['active_messages']}")
    print(f"Context Compressions: {stats['compressions_performed']}")
    print(f"Final Utilization: {stats['utilization']:.1%}")
    print(f"Available Tokens: {stats['available_tokens']:,}")
    
    print("\n💡 Key Takeaways:")
    print("  • The 500-token buffer prevents sudden context overflow")
    print("  • Compression triggers at 90% capacity maintain performance")
    print("  • Most recent messages are always preserved")
    print("  • Dynamic sizing adapts to device capabilities")
    print("=" * 70)

def demonstrate_three_tiered_system():
    """Demonstrate the three-tiered context compression system"""
    
    print("\n" + "=" * 70)
    print("Chapter 11: Three-Tiered Context System Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Create three-tiered system
    config = ContextConfig(
        max_context_tokens=2048,
        buffer_tokens=500,
        system_prompt_tokens=100,
        compression_threshold=0.9
    )
    
    system = ThreeTieredContextSystem(config)
    
    print(f"Configuration:")
    print(f"  Max Context: {config.max_context_tokens} tokens")
    print(f"  Buffer: {config.buffer_tokens} tokens")
    print(f"  Compression Threshold: {config.compression_threshold:.0%}")
    
    # Simulate extended conversation
    print(f"\nSimulating extended conversation with three-tiered compression...")
    
    conversations = [
        ("user", "I'm learning Python programming and need help with functions.", 15),
        ("assistant", "Great! Functions are fundamental in Python. A function is defined using 'def' keyword. Here's a basic example: def greet(name): return f'Hello, {name}!'", 35),
        ("user", "How do I pass multiple arguments to a function?", 12),
        ("assistant", "You can pass multiple arguments in several ways: positional args, keyword args, *args for variable length, and **kwargs for keyword arguments.", 40),
        ("user", "What's the difference between *args and **kwargs?", 10),
        ("assistant", "*args collects extra positional arguments into a tuple, while **kwargs collects extra keyword arguments into a dictionary. Example: def func(*args, **kwargs):", 35),
        ("user", "Can you show me a practical example with both?", 10),
        ("assistant", "Sure! def process_data(name, age, *scores, **options): print(f'{name}, {age}'); print(f'Scores: {scores}'); print(f'Options: {options}'). Call it like: process_data('Alice', 25, 85, 90, 78, verbose=True, format='json')", 60),
        ("user", "How do I handle errors in functions?", 10),
        ("assistant", "Use try-except blocks: def safe_divide(a, b): try: return a / b; except ZeroDivisionError: return 'Cannot divide by zero'; except TypeError: return 'Invalid input types'", 45),
        ("user", "What about function decorators?", 8),
        ("assistant", "Decorators modify function behavior. Example: @timer decorator adds timing. def timer(func): def wrapper(*args, **kwargs): start = time.time(); result = func(*args, **kwargs); print(f'Time: {time.time() - start}s'); return result; return wrapper", 55),
        ("user", "How do I create my own decorator?", 10),
        ("assistant", "Create a function that takes another function and returns a modified version. Example: def log_calls(func): def wrapper(*args, **kwargs): print(f'Calling {func.__name__}'); return func(*args, **kwargs); return wrapper", 50),
        ("user", "Can you show me a complete example with error handling and decorators?", 15),
        ("assistant", "Here's a complete example: @log_calls; def calculate_average(*numbers): try: return sum(numbers) / len(numbers); except ZeroDivisionError: return 0; except TypeError: return 'Invalid input'. This combines decorators, error handling, and *args.", 65),
    ]
    
    for i, (role, content, tokens) in enumerate(conversations, 1):
        print(f"\nTurn {i}: {role} ({tokens} tokens)")
        system.add_message(role, content, tokens)
        
        # Show statistics after each message
        stats = system.get_statistics()
        print(f"  Raw Buffer: {stats['raw_buffer_messages']} messages, {stats['raw_buffer_tokens']} tokens")
        print(f"  Summary Chain: {stats['summary_chain_length']} entries")
        print(f"  Semantic Index: {stats['semantic_index_size']} entities")
        print(f"  Compressions: {stats['compressions_performed']}")
        print(f"  Utilization: {stats['total_capacity_utilization']:.1%}")
    
    # Demonstrate semantic memory retrieval
    print(f"\n🔍 Semantic Memory Retrieval Demo:")
    print("-" * 40)
    
    test_queries = [
        "How do I handle errors?",
        "What are decorators?",
        "Show me function examples",
        "Python programming help"
    ]
    
    for query in test_queries:
        matches = system.get_semantic_matches(query, top_k=3)
        print(f"Query: '{query}'")
        print(f"  Semantic Matches: {matches}")
    
    # Show final prompt construction
    print(f"\n📝 Final Prompt Construction:")
    print("-" * 40)
    
    final_query = "Can you help me with Python functions?"
    final_prompt = system.get_final_prompt(final_query)
    
    print("Final Prompt Structure:")
    print("=" * 50)
    print(final_prompt[:500] + "..." if len(final_prompt) > 500 else final_prompt)
    
    # Final statistics
    print(f"\n📊 Final System Statistics:")
    print("-" * 40)
    final_stats = system.get_statistics()
    print(f"Total Messages Processed: {len(conversations)}")
    print(f"Raw Buffer Messages: {final_stats['raw_buffer_messages']}")
    print(f"Summary Chain Entries: {final_stats['summary_chain_length']}")
    print(f"Semantic Index Size: {final_stats['semantic_index_size']}")
    print(f"Compressions Performed: {final_stats['compressions_performed']}")
    print(f"Final Utilization: {final_stats['total_capacity_utilization']:.1%}")
    
    print("\n💡 Three-Tiered System Benefits:")
    print("  • Tier 1: Immediate coherence with 500-token buffer")
    print("  • Tier 2: Sequential memory via Summary Chain")
    print("  • Tier 3: Non-sequential recall via Semantic Index")
    print("  • Prevents catastrophic forgetting")
    print("  • Maintains long-term context efficiently")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_context_optimization()
    demonstrate_three_tiered_system()

