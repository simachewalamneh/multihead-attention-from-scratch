"""
Part 3 — Multi-Head Causal Self-Attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
  
    def __init__(self, embed_dim: int, n_head: int, max_seq_len: int = 1024):
        super().__init__()
        assert embed_dim % n_head == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_head ({n_head}) "
            f"so it can be split into equal-sized heads."
        )
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
       
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, t: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        t = t.view(B, T, self.n_head, self.head_dim)
        t = t.transpose(1, 2).contiguous()
        return t

    def _merge_heads(self, t: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        t = t.transpose(1, 2).contiguous()
        t = t.view(B, T, self.embed_dim)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        Q = self._split_heads(self.q_proj(x), B, T)   # (B, nh, T, hd)
        K = self._split_heads(self.k_proj(x), B, T)   # (B, nh, T, hd)
        V = self._split_heads(self.v_proj(x), B, T)   # (B, nh, T, hd)
      
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        mask = self.causal_mask[:T, :T]                                # (T, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)                       # (B, nh, T, T)
        head_out = attn_weights @ V                                    # (B, nh, T, hd)

        merged = self._merge_heads(head_out, B, T)                     # (B, T, C)
        out = self.out_proj(merged)                                    # (B, T, C)
        return out
