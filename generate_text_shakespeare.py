"""
Same pipeline as generate_text.py, but trained on TinyShakespeare
(data/tinyshakespeare.txt, ~1.1MB, ~1.1M characters) instead of the
5-sentence toy corpus — enough real data and training steps to see the
model start producing Shakespeare-*shaped* text (character names, line
structure, archaic words), not just memorized fragments of five sentences.

Still a small model trained briefly by real-LLM standards — this is a
demonstration of the pipeline working at meaningfully larger scale, not
a production-quality language model. Expect ~1-3 minutes to train on a
laptop CPU.

Run: python generate_text_shakespeare.py
"""
import os
import time
import torch

from model import GPTLite, beam_search

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Load TinyShakespeare (falls back to a clear error if the file is missing
#    rather than silently training on nothing).
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Expected TinyShakespeare at {DATA_PATH}.\n"
        "Download it with:\n"
        '  curl -sL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" '
        f'-o "{DATA_PATH}"'
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    CORPUS = f.read()

chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)


def encode(s: str) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


def decode(ids) -> str:
    return "".join(itos[int(i)] for i in ids)


data = encode(CORPUS)

# Train/val split — held-out data lets us report a loss the model hasn't
# directly memorized against, a more honest signal than training loss alone.
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

BLOCK_SIZE = 64         # training context window
MAX_SEQ_LEN = 160       # must cover prompt_len + max_new_tokens at generation time
BATCH_SIZE = 32


def get_batch(split: str = "train"):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_val_loss(model, n_batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch("val")
        logits, _ = model(xb, cache_list=None)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# 2. Build and train GPTLite — still small, but real training data at scale.
# ---------------------------------------------------------------------------
model = GPTLite(
    vocab_size=vocab_size,
    embed_dim=64,
    n_head=4,
    n_layer=3,
    max_seq_len=MAX_SEQ_LEN,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

N_STEPS = 2000
print(f"Vocab size: {vocab_size}  |  Corpus length: {len(data):,} chars "
      f"(train: {len(train_data):,}, val: {len(val_data):,})")
print(f"Training GPTLite for {N_STEPS} steps on TinyShakespeare...\n")

model.train()
t0 = time.perf_counter()
for step in range(N_STEPS):
    xb, yb = get_batch("train")
    logits, _ = model(xb, cache_list=None)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0 or step == N_STEPS - 1:
        val_loss = estimate_val_loss(model)
        elapsed = time.perf_counter() - t0
        print(f"  step {step:5d}  train_loss {loss.item():.3f}  val_loss {val_loss:.3f}  ({elapsed:.0f}s elapsed)")

model.eval()


# ---------------------------------------------------------------------------
# 3. Generate with every decoding perspective from the same prompt.
# ---------------------------------------------------------------------------
prompt = "ROMEO:\n"
prompt_ids = encode(prompt).unsqueeze(0)

print(f"\nPrompt: {prompt!r}\n" + "=" * 60)

for strategy, kwargs in [
    ("greedy", {}),
    ("temperature", {"temperature": 0.8}),
    ("temperature", {"temperature": 1.2}),
    ("top_k", {"top_k": 10, "temperature": 1.0}),
    ("top_p", {"top_p": 0.9, "temperature": 1.0}),
]:
    out = model.generate(prompt_ids, max_new_tokens=150, strategy=strategy, **kwargs)
    label = f"{strategy} {kwargs}" if kwargs else strategy
    print(f"\n--- {label} ---\n{decode(out[0])}")

beam_out = beam_search(model, prompt_ids, max_new_tokens=100, beam_width=4)
print(f"\n--- beam_search (width=4) ---\n{decode(beam_out[0])}")


# ---------------------------------------------------------------------------
# 4. KV-cache vs. no-cache speed benchmark.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("KV-cache speed benchmark")
print("=" * 60)

N_NEW = 100

t0 = time.perf_counter()
_ = model.generate(prompt_ids, max_new_tokens=N_NEW, strategy="greedy", use_cache=True)
t_cached = time.perf_counter() - t0

t0 = time.perf_counter()
_ = model.generate(prompt_ids, max_new_tokens=N_NEW, strategy="greedy", use_cache=False)
t_uncached = time.perf_counter() - t0

print(f"  With KV-cache:    {t_cached:.4f}s for {N_NEW} tokens")
print(f"  Without KV-cache: {t_uncached:.4f}s for {N_NEW} tokens")
print(f"  Speedup: {t_uncached / t_cached:.2f}x")
