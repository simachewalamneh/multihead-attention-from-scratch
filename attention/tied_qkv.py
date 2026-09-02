
import math
import torch
import torch.nn as nn

from .multi_head import CausalSelfAttention

class CausalSelfAttentionTiedQKV(nn.Module):
 
    def __init__(self, embed_dim: int, n_head: int, max_seq_len: int = 1024):
        super().__init__()
        assert embed_dim % n_head == 0
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def load_from_separate(self, other: CausalSelfAttention):
        """Copy weights from a CausalSelfAttention so outputs can be compared
        numerically ."""
        #it forces both models to have the exact same numbers in their weights (copying W_q, W_k, W_v into the right row-slices of qkv_proj.weight
        with torch.no_grad():
            self.qkv_proj.weight.copy_(
                torch.cat([other.q_proj.weight, other.k_proj.weight, other.v_proj.weight], dim=0)
            )
            self.out_proj.weight.copy_(other.out_proj.weight)

    def _split_heads(self, t, B, T):
        t = t.view(B, T, self.n_head, self.head_dim)
        return t.transpose(1, 2).contiguous()

    def _merge_heads(self, t, B, T):
        t = t.transpose(1, 2).contiguous()
        return t.view(B, T, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)                 # (B, T, 3C)
        Q, K, V = qkv.split(self.embed_dim, dim=-1)   # each (B, T, C)

        Q = self._split_heads(Q, B, T)
        K = self._split_heads(K, B, T)
        V = self._split_heads(V, B, T)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        head_out = attn_weights @ V

        merged = self._merge_heads(head_out, B, T)
        return self.out_proj(merged)
