"""
End-to-end demo: train GPTLite on a tiny character-level corpus, then
generate text with every decoding strategy from model/sampling.py plus
beam search, so you can directly compare the different perspectives on
the SAME trained model and SAME prompt.

Also benchmarks generation speed with vs. without the KV-cache, to show
concretely why Part 3's bonus (attention/kv_cache.py) matters once
attention is actually embedded in a generation loop instead of a single
forward pass.

Run: python generate_text.py
"""
import time
import torch

from model import GPTLite, beam_search

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Tiny toy corpus (character-level, so the vocabulary is small enough to
#    train from scratch in seconds on CPU — the point is to exercise the
#    full pipeline end-to-end, not to produce a fluent model).
# ---------------------------------------------------------------------------
CORPUS = """
attention lets every word ask who in this sentence should i pay attention to.
causal masking stops it from looking into the future.
multiple heads let it ask that question in several different ways at once.
the query the key and the value are three different views of the same word.
scaled dot product attention computes a score a scale a softmax and a weighted sum.
""".strip()

chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)


def encode(s: str) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


def decode(ids) -> str:
    return "".join(itos[int(i)] for i in ids)


data = encode(CORPUS)
BLOCK_SIZE = 32        # training context window
MAX_SEQ_LEN = 160      # must cover prompt_len + max_new_tokens used at generation time


def get_batch(batch_size: int = 16):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# 2. Build and train GPTLite for a small number of steps — enough to move
#    off random-noise output, not enough to be a serious language model.
# ---------------------------------------------------------------------------
model = GPTLite(
    vocab_size=vocab_size,
    embed_dim=64,
    n_head=4,
    n_layer=3,
    max_seq_len=MAX_SEQ_LEN,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

print(f"Vocab size: {vocab_size}  |  Corpus length: {len(data)} chars")
print("Training GPTLite for 300 steps on the toy corpus...\n")

model.train()
for step in range(300):
    xb, yb = get_batch()
    logits, _ = model(xb, cache_list=None)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), yb.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == 299:
        print(f"  step {step:4d}  loss {loss.item():.3f}")

model.eval()


# ---------------------------------------------------------------------------
# 3. Generate with every decoding perspective from the same prompt.
# ---------------------------------------------------------------------------
prompt = "attention "
prompt_ids = encode(prompt).unsqueeze(0)

print(f"\nPrompt: {prompt!r}\n" + "=" * 60)

for strategy, kwargs in [
    ("greedy", {}),
    ("temperature", {"temperature": 0.8}),
    ("temperature", {"temperature": 1.5}),
    ("top_k", {"top_k": 5, "temperature": 1.0}),
    ("top_p", {"top_p": 0.9, "temperature": 1.0}),
]:
    out = model.generate(prompt_ids, max_new_tokens=60, strategy=strategy, **kwargs)
    label = f"{strategy} {kwargs}" if kwargs else strategy
    print(f"[{label:35s}] {decode(out[0])!r}")

beam_out = beam_search(model, prompt_ids, max_new_tokens=60, beam_width=4)
print(f"[{'beam_search (width=4)':35s}] {decode(beam_out[0])!r}")


# ---------------------------------------------------------------------------
# 4. KV-cache vs. no-cache speed benchmark — same model, same prompt, same
#    number of generated tokens, only the caching strategy differs.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("KV-cache speed benchmark")
print("=" * 60)

N_NEW = 80

t0 = time.perf_counter()
_ = model.generate(prompt_ids, max_new_tokens=N_NEW, strategy="greedy", use_cache=True)
t_cached = time.perf_counter() - t0

t0 = time.perf_counter()
_ = model.generate(prompt_ids, max_new_tokens=N_NEW, strategy="greedy", use_cache=False)
t_uncached = time.perf_counter() - t0

print(f"  With KV-cache:    {t_cached:.4f}s for {N_NEW} tokens")
print(f"  Without KV-cache: {t_uncached:.4f}s for {N_NEW} tokens")
print(f"  Speedup: {t_uncached / t_cached:.2f}x")
print("\n(At this tiny model size the constant-factor overhead of Python-level")
print("loop and tensor-op dispatch dominates, so the speedup is modest here —")
print("the O(T^2) vs O(T) gap this bonus targets becomes dramatic as sequence")
print("length and model size grow, which is exactly why every production LLM")
print("serving stack uses KV-caching.)")
