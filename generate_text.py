import os
import time
import torch

from model import GPTLite, beam_search

torch.manual_seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASETS = {
    "1": {
        "label": "Amharic Ethiopia corpus",
        "path": os.path.join(DATA_DIR, "amharic_ethiopia.txt"),
        "default_prompt": "ኢትዮጵያ ",
        "block_size": 48,
        "max_seq_len": 220,
        "n_steps": 800,
        "use_val_split": False,
    },
    "2": {
        "label": "TinyShakespeare",
        "path": os.path.join(DATA_DIR, "tinyshakespeare.txt"),
        "default_prompt": "ROMEO:\n",
        "max_seq_len": 160,
        "n_steps": 2000,
        "use_val_split": True,
    },
}

def choose_dataset() -> dict:
    print("Choose a training corpus:")
    for key, cfg in DATASETS.items():
        print(f"  {key}. {cfg['label']}")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in DATASETS:
            return DATASETS[choice]
        print("  Please enter 1 or 2.")


dataset_cfg = choose_dataset()

if not os.path.exists(dataset_cfg["path"]):
    raise FileNotFoundError(
        f"Expected {dataset_cfg['label']} at {dataset_cfg['path']}, but it's not there.\n"
        "Make sure the file exists under data/ before running this script."
    )

with open(dataset_cfg["path"], "r", encoding="utf-8") as f:
    CORPUS = f.read().strip()

chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

def encode(s: str) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids) -> str:
    return "".join(itos[int(i)] for i in ids)


data = encode(CORPUS)
BLOCK_SIZE = dataset_cfg["block_size"]
MAX_SEQ_LEN = dataset_cfg["max_seq_len"]
N_STEPS = dataset_cfg["n_steps"]
USE_VAL_SPLIT = dataset_cfg["use_val_split"]

if USE_VAL_SPLIT:
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
else:
    # Corpus too small to meaningfully hold out a validation slice —
    # train on all of it, same as the original toy-corpus demo.
    train_data, val_data = data, data

def get_batch(split: str = "train", batch_size: int = 32):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK_SIZE - 1, (batch_size,))
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

# Build and train GPTLite on the chosen corpus.

model = GPTLite(
    vocab_size=vocab_size,
    embed_dim=64,
    n_head=4,
    n_layer=3,
    max_seq_len=MAX_SEQ_LEN,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

print(f"\nDataset: {dataset_cfg['label']}")
print(f"Vocab size: {vocab_size}  |  Corpus length: {len(data):,} chars")
print(f"Training GPTLite for {N_STEPS} steps...\n")

model.train()
t0 = time.perf_counter()
for step in range(N_STEPS):
    xb, yb = get_batch("train")
    logits, _ = model(xb, cache_list=None)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), yb.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_every = max(N_STEPS // 8, 1)
    if step % log_every == 0 or step == N_STEPS - 1:
        elapsed = time.perf_counter() - t0
        if USE_VAL_SPLIT:
            val_loss = estimate_val_loss(model)
            print(f"  step {step:5d}  train_loss {loss.item():.3f}  val_loss {val_loss:.3f}  ({elapsed:.0f}s elapsed)")
        else:
            print(f"  step {step:5d}  loss {loss.item():.3f}  ({elapsed:.0f}s elapsed)")

model.eval()

# Prompt entered interactively by the user, then generated with every

def read_prompt() -> str:
  
    default = dataset_cfg["default_prompt"]
    while True:
        raw = input(f"\nEnter a prompt (blank = {default!r}): ").strip()
        if raw == "":
            raw = default
        unknown = sorted(set(raw) - set(chars))
        if unknown:
            print(f"  (dropping characters not seen in training: {unknown!r})")
            raw = "".join(c for c in raw if c in stoi)
        if raw:
            return raw
        print("  Prompt was empty after removing unsupported characters — try again.")


prompt = read_prompt()
prompt_ids = encode(prompt).unsqueeze(0)

print(f"\nPrompt: {prompt!r}\n" + "=" * 60)

MAX_NEW_TOKENS = 120 if USE_VAL_SPLIT else 100

for strategy, kwargs in [
    ("greedy", {}),
    ("temperature", {"temperature": 0.8}),
    ("temperature", {"temperature": 1.5}),
    ("top_k", {"top_k": 5, "temperature": 1.0}),
    ("top_p", {"top_p": 0.9, "temperature": 1.0}),
]:
    out = model.generate(prompt_ids, max_new_tokens=MAX_NEW_TOKENS, strategy=strategy, **kwargs)
    label = f"{strategy} {kwargs}" if kwargs else strategy
    print(f"\n--- {label} ---\n{decode(out[0])}")

beam_out = beam_search(model, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, beam_width=4)
print(f"\n--- beam_search (width=4) ---\n{decode(beam_out[0])}")


# KV-cache vs. no-cache speed benchmark — same model, same prompt, same

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