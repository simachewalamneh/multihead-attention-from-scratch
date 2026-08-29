"""
One Transformer decoder block: attention sub-layer + MLP sub-layer, each
wrapped in a residual connection and preceded by LayerNorm (pre-norm, the
standard GPT-2-style arrangement — more stable to train than post-norm for
deep stacks).
"""
import torch
import torch.nn as nn

from attention import CausalSelfAttentionKVCache


class MLP(nn.Module):
    """
    The "feed-forward" sub-layer. Attention mixes information ACROSS
    tokens (each token's new value depends on other tokens); the MLP is
    the sub-layer that processes EACH token independently and gives the
    model non-linear capacity to actually transform what attention
    gathered, not just re-weight and re-sum it. The standard widening
    factor is 4x: project up to 4*embed_dim, apply a non-linearity, project
    back down.
    """

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
    """
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

    Two design choices worth defending:
    - Pre-norm (LayerNorm BEFORE the sub-layer, not after): with post-norm,
      gradients have to flow back through LayerNorm at every block, which
      becomes unstable as depth grows. Pre-norm keeps a clean residual
      "highway" — the raw `x + ...` path never has a LayerNorm sitting in
      it — which is why virtually every modern LLM uses pre-norm.
    - The residual connections themselves (`x = x + ...`, not `x =
      sublayer(x)`) are what let this stack to arbitrary depth in the
      first place: they guarantee that even if a given block's sub-layer
      learns to contribute ~nothing, gradients and information can still
      flow straight through via the `+ x` identity path.
    """

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
