import torch
import torch.nn.functional as F

def greedy(logits: torch.Tensor) -> int:
    #Always take the single highest-probability token.
    return int(torch.argmax(logits).item())

def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
   
   # Divide logits by temperature before softmax, then sample.
   #    
    scaled = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

def top_k_sample(logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> int:
    """
    Zero out every token except the k highest-probability ones, then
    sample from what remains (renormalized).
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
