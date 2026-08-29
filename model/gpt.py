"""
GPTLite — a minimal decoder-only Transformer for text generation, built
entirely on top of the from-scratch attention modules in attention/.

This is the point of the whole assignment applied end-to-end: attention
alone doesn't generate text. It's one ingredient. What actually generates
text is:

    tokens -> [token embedding + positional embedding]
           -> N x TransformerBlock (attention + MLP, each with residuals)
           -> final LayerNorm
           -> linear projection to vocabulary logits
           -> a DECODING STRATEGY that turns logits into a chosen token
           -> feed that token back in, repeat

Everything upstream of "decoding strategy" is what earlier parts of this
project built. Everything from "decoding strategy" onward is what makes
generation feel different depending on how you read the model's output
distribution — see model/sampling.py for four different perspectives on
that last step, and beam_search below for a fifth that operates on whole
sequences instead of one token at a time.
"""
import torch
import torch.nn as nn

from model.block import TransformerBlock
from model.sampling import greedy, temperature_sample, top_k_sample, top_p_sample


class GPTLite(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        n_head: int = 4,
        n_layer: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer

        # Token embedding: learned lookup table, one vector per vocabulary
        # entry. Positional embedding: a SEPARATE learned vector per
        # sequence position, added on top. Attention itself has no notion
        # of order (it's a weighted sum over a SET of tokens) — without
        # positional information, "the cat sat" and "sat the cat" would
        # produce identical attention patterns. Positional embeddings are
        # what inject "where in the sequence am I" into each token.
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_head, max_seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: the input token-embedding matrix and the output
        # vocabulary-projection matrix are forced to be the SAME weights
        # (transposed). Intuition: "the vector that represents token X
        # when reading it in" and "the vector that says how much the
        # model's current belief looks like token X when writing it out"
        # are naturally the same kind of object, and tying them roughly
        # halves the embedding parameter count with usually little to no
        # quality loss. (This is a different weight-tying than the Q/K/V
        # tying in attention/tied_qkv.py — same NAME, different tensors
        # being tied, worth being precise about the distinction if asked.)
        self.head.weight = self.token_embed.weight

    def forward(self, idx: torch.Tensor, cache_list: list | None = None):
        """
        idx: (B, T) token ids.
        cache_list: None, or a list of per-block KV caches (for
            incremental decoding — see generate()).
        Returns (logits, new_cache_list) where logits is (B, T, vocab_size).
        """
        B, T = idx.shape
        device = idx.device

        # Positions must account for how much history is already cached:
        # if we're mid-generation with a cache of length T_cache, the new
        # token(s) occupy absolute positions [T_cache, T_cache + T).
        T_cache = cache_list[0]["K"].shape[2] if cache_list is not None else 0
        positions = torch.arange(T_cache, T_cache + T, device=device)

        x = self.token_embed(idx) + self.pos_embed(positions)   # (B, T, C)
        x = self.dropout(x)

        new_cache_list = []
        for i, block in enumerate(self.blocks):
            block_cache = cache_list[i] if cache_list is not None else None
            x, updated_cache = block(x, cache=block_cache)
            new_cache_list.append(updated_cache)

        x = self.ln_final(x)
        logits = self.head(x)                                   # (B, T, vocab_size)
        return logits, new_cache_list

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        strategy: str = "temperature",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation: repeatedly predict the next token and
        append it, one perspective (strategy) at a time.

        strategy in {"greedy", "temperature", "top_k", "top_p"}.

        use_cache=True routes through the KV-cache path (attention/kv_cache.py)
        so each new token only costs O(T) instead of O(T^2) recomputation —
        see benchmark_kv_cache_speed() in generate_text.py for a direct
        timed comparison against use_cache=False.
        """
        self.eval()
        strategy_fn = {
            "greedy": lambda logits: greedy(logits),
            "temperature": lambda logits: temperature_sample(logits, temperature),
            "top_k": lambda logits: top_k_sample(logits, top_k, temperature),
            "top_p": lambda logits: top_p_sample(logits, top_p, temperature),
        }[strategy]

        cache_list = None
        generated = idx

        if use_cache:
            # First pass: feed the whole prompt, build the initial cache.
            logits, cache_list = self.forward(generated, cache_list=None)
            for _ in range(max_new_tokens):
                next_id = strategy_fn(logits[0, -1])
                next_tensor = torch.tensor([[next_id]], device=idx.device)
                generated = torch.cat([generated, next_tensor], dim=1)
                # Only the NEW token goes through forward now — the cache
                # already holds everything before it.
                logits, cache_list = self.forward(next_tensor, cache_list=cache_list)
        else:
            for _ in range(max_new_tokens):
                # No cache: recompute attention over the full sequence so
                # far, every single step (the O(T^2) baseline).
                cropped = generated[:, -self.max_seq_len:]
                logits, _ = self.forward(cropped, cache_list=None)
                next_id = strategy_fn(logits[0, -1])
                next_tensor = torch.tensor([[next_id]], device=idx.device)
                generated = torch.cat([generated, next_tensor], dim=1)

        return generated


@torch.no_grad()
def beam_search(model: GPTLite, idx: torch.Tensor, max_new_tokens: int, beam_width: int = 4) -> torch.Tensor:
    """
    A fifth perspective on decoding, operating on WHOLE SEQUENCES rather
    than one token at a time: track the `beam_width` highest cumulative
    log-probability sequences at every step, expand each by one token,
    and keep only the best `beam_width` survivors.

    Perspective: greedy commits irrevocably to the single best token at
    every step, which can lock out a sequence that would have been better
    OVERALL but required a locally-worse first choice. Beam search hedges
    against that by keeping several candidate continuations alive at
    once. It trades more compute (beam_width times the forward passes)
    for a better ARGMAX-style approximation of the single most likely
    whole sequence — note this makes it a fundamentally different
    perspective from the sampling-based strategies above: beam search is
    still deterministic and still trying to maximize likelihood, just
    over sequences instead of single tokens.
    """
    model.eval()
    device = idx.device

    # Each beam: (token_ids tensor, cumulative log-prob)
    beams = [(idx, 0.0)]

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            logits, _ = model.forward(seq, cache_list=None)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            for lp, tok in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tok]], device=device)], dim=1)
                candidates.append((new_seq, score + lp))

        # Keep only the top beam_width candidates across ALL beams'
        # expansions (not top-k per beam) — this is what lets a single
        # strong beam dominate, or several mediocre beams get pruned.
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:beam_width]

    best_seq, best_score = max(beams, key=lambda c: c[1])
    return best_seq
