import torch
import torch.nn as nn  #importing pytourch Nural net. component

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
    
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_head, max_seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight 
      
    def forward(self, idx: torch.Tensor, cache_list: list | None = None):
       
        B, T = idx.shape
        device = idx.device

        T_cache = cache_list[0]["K"].shape[2] if cache_list is not None else 0
        positions = torch.arange(T_cache, T_cache + T, device=device)

        x = self.token_embed(idx) + self.pos_embed(positions)   # (B, T, C)
        x = self.dropout(x)

        new_cache_list = []
        for i, block in enumerate(self.blocks):
            block_cache = cache_list[i] if cache_list is not None else None
            x, updated_cache = block(x, cache=block_cache) #gateway into the actual Transformer mathematics
            new_cache_list.append(updated_cache)

        x = self.ln_final(x)
        logits = self.head(x)                                   # (B, T, vocab_size)
        return logits, new_cache_list

    @torch.no_grad() #prevents PyTorch from building the backward computation graph.
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
    
        self.eval() #switch the model in to evaluation mode
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
                logits, cache_list = self.forward(next_tensor, cache_list=cache_list)
        else:
            for _ in range(max_new_tokens):
                cropped = generated[:, -self.max_seq_len:]
                logits, _ = self.forward(cropped, cache_list=None)
                next_id = strategy_fn(logits[0, -1])
                next_tensor = torch.tensor([[next_id]], device=idx.device)
                generated = torch.cat([generated, next_tensor], dim=1)

        return generated

@torch.no_grad()
def beam_search(model: GPTLite, idx: torch.Tensor, max_new_tokens: int, beam_width: int = 4) -> torch.Tensor:
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
    
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:beam_width]

    best_seq, best_score = max(beams, key=lambda c: c[1])
    return best_seq
