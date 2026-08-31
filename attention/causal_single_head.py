"""
Part 2 — Causal Masking on top of single-head attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSingleHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, head_dim: int | None = None, max_seq_len: int = 1024):
        super().__init__()
        self.head_dim = head_dim if head_dim is not None else embed_dim
        self.q_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, T, T)
   
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)                        # rows still sum to 1.0
        out = attn_weights @ V
        return out
