"""
Part 3 — Multi-Head Causal Self-Attention.

This is the module the text-generation model (see model/) actually uses
as its attention layer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Splits the embedding dimension into n_head independent heads, runs
    scaled dot-product causal attention in each head IN PARALLEL (as one
    batched tensor op, not a Python for-loop over heads), then concatenates
    and projects back.

    Design choices worth defending:
    - Q, K, V are still produced with ONE big nn.Linear(C, C) each (not
      n_head separate small Linears). This is mathematically identical to
      n_head independent (C -> head_dim) projections stacked together;
      slicing the OUTPUT into heads is equivalent to, and cheaper than,
      n_head separate layers.
    - The reshape/transpose dance (B,T,C) -> (B,T,nh,hd) -> (B,nh,T,hd) is
      the single most bug-prone line in the whole assignment. Only
      transposing dims 1 and 2 (T and n_head) is correct: it moves n_head
      next to the batch dim so every subsequent matmul is a BATCHED matmul
      over (B, n_head) independent (T, head_dim) attention problems. Get
      the transpose dims wrong (e.g. reshape directly to (B, nh, T, hd)
      without the intermediate view+transpose) and you silently scramble
      which head_dim slice belongs to which token — the tensor still
      "runs" because shapes still match, but the numbers are wrong. See
      demonstrate_scrambling_bug.py for a worked example.
    - .contiguous() is called after every .transpose(): transpose returns a
      VIEW with non-contiguous strides. .view() requires contiguous
      memory, so we call .contiguous() before any .view() that follows a
      transpose. Skipping this either raises a RuntimeError or, worse,
      silently misbehaves.
    """

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
        # The output projection AFTER heads are concatenated. Without this,
        # the model could only ever produce a simple concatenation of head
        # outputs; this layer lets it learn how to blend information
        # across heads, which is what makes heads cooperate rather than
        # just sit side by side.
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

        # Batched matmul: for every (batch, head) pair independently,
        # (T, hd) @ (hd, T) -> (T, T). PyTorch broadcasts matmul over all
        # leading dims, so this one line runs all heads and all batch items
        # in parallel with no Python loop.
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        mask = self.causal_mask[:T, :T]                                # (T, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)                       # (B, nh, T, T)
        head_out = attn_weights @ V                                    # (B, nh, T, hd)

        merged = self._merge_heads(head_out, B, T)                     # (B, T, C)
        out = self.out_proj(merged)                                    # (B, T, C)
        return out
