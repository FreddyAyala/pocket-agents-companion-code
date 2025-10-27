"""
Chapter 8: Architectural Optimizations - Complete Implementations

This module provides complete, production-ready implementations of the architectural
innovations discussed in Chapter 8: KV Cache, Grouped Query Attention (GQA), and
Sliding Window Attention (SWA).

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Optional


class OptimizedKVCache:
    """
    Efficient Key-Value cache for autoregressive generation.
    
    The KV cache is critical for performance, transforming O(n²) complexity
    to O(n) by caching intermediate attention states.
    """
    
    def __init__(
        self, 
        max_length: int, 
        num_heads: int, 
        head_dim: int, 
        dtype: torch.dtype = torch.float16,
        device: str = 'auto'
    ):
        """
        Initialize the KV cache.
        
        Args:
            max_length: Maximum sequence length to cache
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for cache (float16 for efficiency)
            device: Device to store cache on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.max_length = max_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # Auto-detect best available device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Pre-allocate cache tensors for maximum efficiency
        self.k_cache = torch.zeros(
            (max_length, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (max_length, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        self.current_length = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """
        Update cache with new key-value pairs.
        
        Args:
            k: Key tensor [num_heads, head_dim]
            v: Value tensor [num_heads, head_dim]
            position: Position in sequence to update
        """
        # Handle cache overflow with FIFO strategy
        if position >= self.max_length:
            # Shift cache left to make room
            self.k_cache[:-1] = self.k_cache[1:].clone()
            self.v_cache[:-1] = self.v_cache[1:].clone()
            position = self.max_length - 1
        
        # Update cache at position
        self.k_cache[position] = k
        self.v_cache[position] = v
        
        self.current_length = max(self.current_length, position + 1)
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all cached key-value pairs up to current length.
        
        Returns:
            Tuple of (keys, values) tensors
        """
        return (
            self.k_cache[:self.current_length],
            self.v_cache[:self.current_length]
        )
    
    def clear(self):
        """Clear the cache and reset length."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_length = 0
    
    def memory_usage_mb(self) -> float:
        """Calculate memory usage in MB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        total_elements = 2 * self.max_length * self.num_heads * self.head_dim
        return (total_elements * bytes_per_element) / (1024 * 1024)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    
    GQA reduces memory usage by sharing Key/Value heads across multiple Query heads.
    This is a key technique for efficient on-device inference.
    
    Example: 32 query heads with 8 KV heads = 4x memory reduction on KV cache
    """
    
    def __init__(
        self, 
        embed_dim: int,
        num_heads: int, 
        num_kv_heads: int, 
        dropout: float = 0.0
    ):
        """
        Initialize GQA layer.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (must divide num_heads evenly)
            dropout: Dropout probability
        """
        super().__init__()
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (full number of heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Key and Value projections (reduced number of heads)
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            kv_cache: Optional KV cache for autoregressive generation
            position: Current position in sequence (for caching)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [batch, seq, embed_dim]
        k = self.k_proj(x)  # [batch, seq, kv_dim]
        v = self.v_proj(x)  # [batch, seq, kv_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Repeat KV heads to match query heads (grouped attention)
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)
        
        # Handle KV cache for autoregressive generation
        if kv_cache is not None and seq_len == 1:
            # Update cache with current token
            kv_cache.update(k.squeeze(1), v.squeeze(1), position)
            # Get full history from cache
            k, v = kv_cache.get_cache()
            # Add batch dimension back
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        
        # Compute attention
        attn_output = self._compute_attention(q, k, v, attention_mask)
        
        # Project output
        output = self.out_proj(attn_output)
        
        return output
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        # Transpose for batch matrix multiplication
        q = q.transpose(1, 2)  # [batch, heads, seq_q, head_dim]
        k = k.transpose(1, 2)  # [batch, heads, seq_k, head_dim]
        v = v.transpose(1, 2)  # [batch, heads, seq_v, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask for autoregressive generation
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)
        
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device),
            diagonal=1 + (seq_len_k - seq_len_q)
        ).bool()
        
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to [batch, seq, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        
        return attn_output
    
    def memory_savings_vs_mha(self) -> float:
        """Calculate memory savings compared to Multi-Head Attention."""
        mha_kv_size = 2 * self.num_heads * self.head_dim
        gqa_kv_size = 2 * self.num_kv_heads * self.head_dim
        savings = 1.0 - (gqa_kv_size / mha_kv_size)
        return savings


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention (SWA) implementation.
    
    SWA enables infinite context length with O(n) complexity by only attending
    to a fixed window of recent tokens. Used in models like Mistral.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 1024,
        dropout: float = 0.0
    ):
        """
        Initialize SWA layer.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            window_size: Size of the sliding attention window
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with sliding window attention.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention with sliding window
        attn_output = self._compute_sliding_window_attention(q, k, v, attention_mask)
        
        # Project output
        output = self.out_proj(attn_output)
        
        return output
    
    def _compute_sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention with sliding window constraint."""
        # Transpose for batch matrix multiplication
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create sliding window mask
        seq_len = q.size(2)
        window_mask = self._create_sliding_window_mask(seq_len, q.device)
        
        scores = scores.masked_fill(window_mask, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        
        return attn_output
    
    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sliding window mask.
        
        Returns a mask where True indicates positions that should be masked out.
        """
        # Create causal mask (upper triangular)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        # Create sliding window constraint (lower triangular beyond window)
        window_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=-self.window_size
        )
        
        # Combine: mask if causal OR beyond window
        return causal_mask | window_mask


# ============================================================================
# Benchmarking Functions
# ============================================================================

def benchmark_kv_cache():
    """Benchmark KV cache performance."""
    print("=" * 70)
    print("Benchmarking KV Cache Performance")
    print("=" * 70)
    
    max_length = 2048
    num_heads = 32
    head_dim = 128
    
    # Create cache
    cache = OptimizedKVCache(max_length, num_heads, head_dim)
    
    print(f"\nCache Configuration:")
    print(f"  Max Length: {max_length}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Head Dim: {head_dim}")
    print(f"  Memory Usage: {cache.memory_usage_mb():.2f} MB")
    
    # Benchmark update operations
    num_updates = 1000
    k = torch.randn(num_heads, head_dim, device=cache.device, dtype=cache.dtype)
    v = torch.randn(num_heads, head_dim, device=cache.device, dtype=cache.dtype)
    
    start_time = time.time()
    for i in range(num_updates):
        cache.update(k, v, i % max_length)
    end_time = time.time()
    
    update_time = (end_time - start_time) / num_updates * 1000
    print(f"\nPerformance:")
    print(f"  Avg Update Time: {update_time:.4f} ms")
    print(f"  Updates/sec: {1000 / update_time:.0f}")


def benchmark_gqa_vs_mha():
    """Benchmark GQA vs standard Multi-Head Attention."""
    print("\n" + "=" * 70)
    print("Benchmarking GQA vs Multi-Head Attention")
    print("=" * 70)
    
    embed_dim = 4096
    num_heads = 32
    batch_size = 1
    seq_len = 128
    
    # Standard MHA (32 KV heads)
    mha = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=32)
    
    # GQA (8 KV heads = 4x reduction)
    gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=8)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nConfiguration:")
    print(f"  Embed Dim: {embed_dim}")
    print(f"  Num Heads: {num_heads}")
    print(f"  MHA KV Heads: 32")
    print(f"  GQA KV Heads: 8")
    print(f"  Memory Savings: {gqa.memory_savings_vs_mha() * 100:.1f}%")
    
    # Benchmark inference
    num_runs = 100
    
    # MHA
    start_time = time.time()
    for _ in range(num_runs):
        _ = mha(x)
    mha_time = (time.time() - start_time) / num_runs * 1000
    
    # GQA
    start_time = time.time()
    for _ in range(num_runs):
        _ = gqa(x)
    gqa_time = (time.time() - start_time) / num_runs * 1000
    
    print(f"\nPerformance (avg over {num_runs} runs):")
    print(f"  MHA: {mha_time:.2f} ms")
    print(f"  GQA: {gqa_time:.2f} ms")
    print(f"  Speedup: {mha_time / gqa_time:.2f}x")


if __name__ == "__main__":
    print("\nChapter 8: Architectural Optimizations - Complete Demo")
    print("=" * 70)
    
    # Run benchmarks
    benchmark_kv_cache()
    benchmark_gqa_vs_mha()
    
    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)

