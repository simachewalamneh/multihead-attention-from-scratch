"""
Bonus — KV-Caching for incremental (one-token-at-a-time) decoding.

This is what makes text generation in model/gpt.py fast: without it,
generating token T+1 means recomputing attention over all T previous
tokens from scratch every single step (O(T^2) total work per generated
token). With caching, each new step only computes K/V for the ONE new
token and reuses everything before it (O(T) amortized).
"""
import math
import torch

from .multi_head import CausalSelfAttention


class CausalSelfAttentionKVCache(CausalSelfAttention):
    """
    Adds an incremental-decoding path on top of CausalSelfAttention,
    without changing any of the training-time (full-sequence) behaviour.

    Why KV-caching is valid at all: at generation step t, token t's Query
    only ever attends to Keys/Values of tokens 0..t (causal masking). Those
    K/V vectors for tokens 0..t-1 were already computed in previous steps
    and never change (K/V depend only on the token's own embedding and the
    fixed, already-trained k_proj/v_proj weights — not on what comes after
    it). So instead of recomputing K/V for the whole prefix on every new
    token, we compute K/V once for the new token only, and concatenate it
    onto a running cache. Q, by contrast, is never cached — we only ever
    need the Query of the CURRENT token being generated.
    """

    def forward(self, x: torch.Tensor, cache: dict | None = None):
        """
        x: (B, T_new, C) — typically T_new == 1 during incremental
           decoding, but this also works with T_new == full prompt length
           on the first call.
        cache: None on the very first call, otherwise a dict with 'K' and
           'V' of shape (B, n_head, T_so_far, head_dim) from all previous
           steps.
        Returns (output, new_cache).
        """
        B, T_new, C = x.shape

        Q_new = self._split_heads(self.q_proj(x), B, T_new)   # (B, nh, T_new, hd)
        K_new = self._split_heads(self.k_proj(x), B, T_new)   # (B, nh, T_new, hd)
        V_new = self._split_heads(self.v_proj(x), B, T_new)   # (B, nh, T_new, hd)

        if cache is not None:
            K = torch.cat([cache["K"], K_new], dim=2)   # concat along the T axis
            V = torch.cat([cache["V"], V_new], dim=2)
        else:
            K, V = K_new, V_new

        T_total = K.shape[2]

        # Q only covers the new tokens (rows), but K/V cover the whole
        # history (columns) -> scores is (T_new, T_total), not (T_new, T_new).
        scores = (Q_new @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T_new, T_total)

        # The mask must be sliced accordingly: query row i (which is really
        # absolute position T_total - T_new + i) may see key column j iff
        # j <= T_total - T_new + i. Slicing the precomputed full mask at
        # [T_total - T_new : T_total, :T_total] gives exactly that.
        mask = self.causal_mask[T_total - T_new:T_total, :T_total]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        head_out = attn_weights @ V                                        # (B, nh, T_new, hd)

        merged = self._merge_heads(head_out, B, T_new)
        out = self.out_proj(merged)

        new_cache = {"K": K, "V": V}
        return out, new_cache
