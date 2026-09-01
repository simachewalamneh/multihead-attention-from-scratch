import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int | None = None):
        super().__init__()
        self.head_dim = head_dim if head_dim is not None else embed_dim
        # bias=False mirrors the standard Transformer projections; the bias
        # term adds nothing here because the values are about to be scaled,
        self.q_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        Q = self.q_proj(x)          # (B, T, head_dim)
        K = self.k_proj(x)          # (B, T, head_dim)
        V = self.v_proj(x)          # (B, T, head_dim)
    
        scores = Q @ K.transpose(-2, -1)                      # (B, T, T)

        scores = scores / math.sqrt(self.head_dim)
        # dim=-1 = softmax over the "key" axis, so EVERY ROW sums to 1.0:
        attn_weights = F.softmax(scores, dim=-1)               # (B, T, T)
        out = attn_weights @ V                                  # (B, T, head_dim)
        return out
