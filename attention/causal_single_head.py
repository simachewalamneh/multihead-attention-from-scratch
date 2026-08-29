"""
Part 2 — Causal Masking on top of single-head attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSingleHeadAttention(nn.Module):
    """
    Same as single_head.SingleHeadAttention, but token i is mathematically
    forbidden from seeing tokens j > i.

    Design choice: the mask is applied to the SCORES, before softmax, not
    to the attention weights after softmax. If you zeroed out post-softmax
    weights instead, the remaining (legal) weights would no longer sum to
    1.0 - you'd need a second renormalization pass, and worse, the model
    could still learn a gradient signal that "future position exists and
    was zeroed", which is a subtle leak. Masking pre-softmax with -inf
    avoids all of that in one step, because softmax(-inf) is *exactly* 0
    by construction and doesn't perturb the rest of the row's sum.
    """

    def __init__(self, embed_dim: int, head_dim: int | None = None, max_seq_len: int = 1024):
        super().__init__()
        self.head_dim = head_dim if head_dim is not None else embed_dim
        self.q_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        # Precompute the mask once at construction time rather than every
        # forward call. register_buffer means it moves with .to(device) and
        # gets saved in state_dict (as non-trainable) but is not a Parameter
        # (no gradient, never updated by the optimizer).
        # torch.tril keeps the lower triangle (i >= j) and zeros the rest,
        # so `causal_mask[i, j] == 1` means "position i is ALLOWED to see j".
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, T, T)

        # Slice the precomputed (max_seq_len, max_seq_len) mask down to the
        # actual sequence length T. Broadcasting handles the batch dim.
        mask = self.causal_mask[:T, :T]                                 # (T, T) bool
        # masked_fill fills every position where mask==False (i.e. j > i,
        # "the future") with -inf. -inf, not some very negative number,
        # because we want EXACTLY zero probability, not "very small" —
        # see the docstring above for why that distinction matters.
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)                        # rows still sum to 1.0
        out = attn_weights @ V
        return out
