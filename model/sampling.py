"""
Decoding strategies — different PERSPECTIVES on turning next-token logits
into an actual choice of token. The attention/transformer machinery only
ever produces a probability distribution over the vocabulary; everything
below is about how you read that distribution.

Each function takes raw logits for the NEXT token, shape (vocab_size,),
and returns a single chosen token id (a Python int), except beam_search
which operates at the sequence level and is defined in gpt.py where it has
access to the model's forward pass.
"""
import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> int:
    """
    Always take the single highest-probability token.

    Perspective: maximizes per-step likelihood. Deterministic — same
    prompt always produces the same continuation. Prone to repetition
    loops in practice, because once the model starts a repeated phrase,
    the repeated continuation is often ALSO the highest-probability
    choice (a well-known failure mode of greedy decoding in language
    models).
    """
    return int(torch.argmax(logits).item())


def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Divide logits by temperature before softmax, then sample.

    Perspective: controls how "confident" the distribution is allowed to
    look before you sample from it, without changing WHICH tokens are
    possible. temperature < 1.0 sharpens the distribution (more like
    greedy, less random); temperature > 1.0 flattens it (more random,
    more diverse, more likely to go off the rails). temperature -> 0 is
    mathematically equivalent to greedy; this function IS the general
    case that greedy() is a special case of.
    """
    scaled = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def top_k_sample(logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> int:
    """
    Zero out every token except the k highest-probability ones, then
    sample from what remains (renormalized).

    Perspective: caps the "risk" of sampling — no matter how flat the
    tail of the distribution is, you can never sample a token outside
    the top k. This directly guards against the failure mode of plain
    temperature sampling, where a long flat tail of thousands of
    low-probability tokens can still collectively have enough mass to
    occasionally get sampled and produce nonsense.
    """
    k = min(k, logits.shape[-1])
    scaled = logits / max(temperature, 1e-8)
    top_values, top_indices = torch.topk(scaled, k)
    probs = F.softmax(top_values, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return int(top_indices[choice].item())


def top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Nucleus sampling: sort tokens by probability, keep adding them to the
    candidate set until their cumulative probability just exceeds p, then
    sample from that set (renormalized).

    Perspective: unlike top-k's FIXED cutoff count, top-p's cutoff count
    is DYNAMIC — it adapts to how peaked or flat the distribution is at
    each step. When the model is very confident (one token dominates),
    the nucleus might be just 1-2 tokens; when it's genuinely uncertain
    across many plausible continuations, the nucleus naturally grows to
    include more of them. This is generally considered a better model of
    "how much am I actually unsure about" than a fixed top-k.
    """
    scaled = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep the smallest prefix whose cumulative probability >= p.
    # We always keep at least 1 token, even if its own probability
    # already exceeds p.
    cutoff = torch.searchsorted(cumulative, torch.tensor(p)).item() + 1
    cutoff = max(cutoff, 1)

    nucleus_probs = sorted_probs[:cutoff]
    nucleus_indices = sorted_indices[:cutoff]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()   # renormalize

    choice = torch.multinomial(nucleus_probs, num_samples=1)
    return int(nucleus_indices[choice].item())
