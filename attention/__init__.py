"""
attention — Multi-Head Attention from Scratch (iCog Labs assignment).

    from attention import (
        SingleHeadAttention,          # Part 1
        CausalSingleHeadAttention,    # Part 2
        CausalSelfAttention,          # Part 3
        CausalSelfAttentionKVCache,   # Bonus: incremental decoding
        CausalSelfAttentionTiedQKV,   # Bonus: fused QKV projection
    )

Built using only nn.Linear and raw tensor operations (matmul, softmax,
tril, view/transpose/reshape). No nn.MultiheadAttention, no
F.scaled_dot_product_attention, no einops — anywhere in this package.
"""
from .single_head import SingleHeadAttention
from .causal_single_head import CausalSingleHeadAttention
from .multi_head import CausalSelfAttention
from .kv_cache import CausalSelfAttentionKVCache
from .tied_qkv import CausalSelfAttentionTiedQKV

__all__ = [
    "SingleHeadAttention",
    "CausalSingleHeadAttention",
    "CausalSelfAttention",
    "CausalSelfAttentionKVCache",
    "CausalSelfAttentionTiedQKV",
]
