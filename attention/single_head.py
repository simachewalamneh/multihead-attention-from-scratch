"""
Part 1 — Single-Head Scaled Dot-Product Attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    """
    Takes a batch of token embeddings (B, T, C) and returns an attention
    output of the same shape, computed with ONE query/key/value head.

    Design choices worth defending:
    - Q, K, V come from three SEPARATE nn.Linear layers. They all read the
      same input embedding, but each learns a different projection, so the
      same word produces three different vectors ("what am I looking for",
      "what do I offer", "what do I actually contain"). Tying them would
      collapse three distinct roles into one (see attention.tied_qkv for
      what tying Q/K/V into ONE layer looks like instead — that's a
      *refactor* of this same computation, not a different one).
    - head_dim, not embed_dim, is what we scale by. In the single-head case
      head_dim == embed_dim, but writing it as head_dim makes the class
      trivially reusable as the inner loop of the multi-head version.
    """

    def __init__(self, embed_dim: int, head_dim: int | None = None):
        super().__init__()
        self.head_dim = head_dim if head_dim is not None else embed_dim
        # bias=False mirrors the standard Transformer projections; the bias
        # term adds nothing here because the values are about to be scaled,
        # softmaxed, and blended — the original papers omit it too.
        self.q_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        Q = self.q_proj(x)          # (B, T, head_dim)
        K = self.k_proj(x)          # (B, T, head_dim)
        V = self.v_proj(x)          # (B, T, head_dim)

        # --- Score: Q @ K^T -------------------------------------------------
        # transpose(-2, -1) swaps the last two dims: (B, T, head_dim) ->
        # (B, head_dim, T). We transpose K, not Q, because we want an
        # (T, T) grid: row i, col j = how much token i's query matches
        # token j's key. Q @ K^T gives exactly (B, T, T).
        scores = Q @ K.transpose(-2, -1)                       # (B, T, T)

        # --- Scale: divide by sqrt(head_dim) --------------------------------
        # Why this specific number: assume Q and K entries are i.i.d. with
        # mean 0, variance 1. Each entry of Q @ K^T is a sum of head_dim
        # independent products, so its variance grows linearly with
        # head_dim -> std grows with sqrt(head_dim). Dividing by
        # sqrt(head_dim) renormalizes the score variance back to ~1
        # regardless of head_dim, which keeps softmax's input in a range
        # where its gradient isn't saturated.
        scores = scores / math.sqrt(self.head_dim)

        # --- Softmax across each row -----------------------------------------
        # dim=-1 = softmax over the "key" axis, so EVERY ROW sums to 1.0:
        # for a fixed query token i, its attention weights over all key
        # tokens j form a probability distribution.
        attn_weights = F.softmax(scores, dim=-1)               # (B, T, T)

        # --- Weighted sum: weights @ V ----------------------------------------
        out = attn_weights @ V                                  # (B, T, head_dim)
        return out
