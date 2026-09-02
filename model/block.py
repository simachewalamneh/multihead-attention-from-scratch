import torch
import torch.nn as nn

from attention import CausalSelfAttentionKVCache

class MLP(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc_in = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.fc_out = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.gelu(x)
        x = self.fc_out(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttentionKVCache(embed_dim, n_head, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x: torch.Tensor, cache: dict | None = None):
        attn_out, new_cache = self.attn(self.ln1(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache
