"""
End-to-end demo: train GPTLite on a character-level corpus, then
generate text with every decoding strategy from model/sampling.py plus
beam search, so you can directly compare the different perspectives on
the SAME trained model and SAME prompt — with the prompt entered by you
interactively at the end.

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
# 1. Corpus (character-level, so the vocabulary is small enough to train
#    from scratch in a couple minutes on CPU — the point is to exercise
#    the full pipeline end-to-end, not to produce a fluent model).
# ---------------------------------------------------------------------------
CORPUS = """
Ethiopia is a landlocked country in the Horn of Africa, bordered by Eritrea,
Djibouti, Somalia, Kenya, South Sudan, and Sudan. It is one of the oldest
independent nations in the world and the only African country never
colonized by a European power, aside from a brief Italian occupation
between 1936 and 1941. Ethiopia's history stretches back thousands of
years, from the ancient Kingdom of Aksum, which minted its own coinage and
traded across the Red Sea, to the medieval rock-hewn churches of Lalibela,
carved directly out of solid volcanic rock and still active places of
worship today. The country uses its own calendar, roughly seven to eight
years behind the Gregorian calendar, and its own time system, and it is
often described as having thirteen months of sunshine.

Addis Ababa, the capital, sits at high altitude in the central highlands
and serves as the diplomatic capital of Africa, hosting the headquarters
of the African Union. Ethiopia is remarkably diverse linguistically, with
more than eighty languages spoken across its regions. Amharic has long
served as the federal working language, written in the distinctive Ge'ez
script, while Oromo, Tigrinya, Somali, and many other languages are spoken
widely across different regional states. This diversity extends to
ethnicity, religion, and culture, with the Oromo, Amhara, Tigray, Somali,
and Sidama among the largest of more than eighty recognized ethnic groups.

Ethiopia is widely regarded as the birthplace of coffee, with legend
tracing its discovery to a goat herder named Kaldi in the Kaffa region,
whose goats became energetic after eating coffee cherries. The Ethiopian
coffee ceremony remains a central part of social and family life, involving
roasting green beans over hot coals, grinding them by hand, and brewing
the coffee in a clay pot called a jebena, often accompanied by incense.

Geographically, Ethiopia contains dramatic contrasts: the Simien Mountains
rise to jagged peaks above three thousand meters, home to endemic wildlife
such as the walia ibex and gelada baboon, while the Danakil Depression in
the northeast is one of the hottest and lowest places on Earth, with
active volcanoes, sulfur springs, and salt flats. The Blue Nile, known
locally as the Abay, originates from Lake Tana and carves a dramatic gorge
before crossing into Sudan, eventually joining the White Nile in Khartoum.

Ethiopian cuisine centers on injera, a spongy sourdough flatbread made
from teff flour, served with an array of stews called wat, made with
lentils, vegetables, or meat, and seasoned with berbere spice blends.
Meals are traditionally eaten communally by hand from a shared platter.
Ethiopian Orthodox Christianity, one of the oldest Christian traditions in
the world, coexists with Islam, Protestantism, and traditional beliefs
across the country, and religious festivals such as Timket and Meskel draw
large public celebrations. Ethiopia's long-distance runners, including
Abebe Bikila, Haile Gebrselassie, and Tirunesh Dibaba, have made the
country a dominant force in international athletics for decades.
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
BLOCK_SIZE = 48        # training context window
MAX_SEQ_LEN = 220      # must cover prompt_len + max_new_tokens used at generation time


def get_batch(batch_size: int = 32):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# 2. Build and train GPTLite — this corpus is bigger than the original
#    5-sentence toy version, so it gets more steps to actually fit it.
# ---------------------------------------------------------------------------
model = GPTLite(
    vocab_size=vocab_size,
    embed_dim=64,
    n_head=4,
    n_layer=3,
    max_seq_len=MAX_SEQ_LEN,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

N_STEPS = 800
print(f"Vocab size: {vocab_size}  |  Corpus length: {len(data)} chars")
print(f"Training GPTLite for {N_STEPS} steps on the Ethiopia corpus...\n")

model.train()
for step in range(N_STEPS):
    xb, yb = get_batch()
    logits, _ = model(xb, cache_list=None)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), yb.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0 or step == N_STEPS - 1:
        print(f"  step {step:4d}  loss {loss.item():.3f}")

model.eval()


# ---------------------------------------------------------------------------
# 3. Prompt entered interactively by the user, then generated with every
#    decoding perspective from model/sampling.py so you can compare them
#    side by side on the SAME prompt.
# ---------------------------------------------------------------------------
def read_prompt() -> str:
    """
    Ask the user for a prompt, and silently drop any character the model's
    vocabulary has never seen (the model can only ever produce characters
    it saw during training, so an unseen character has no embedding to
    look up at all).
    """
    while True:
        raw = input("\nEnter a prompt (blank = 'Ethiopia is '): ").strip()
        if raw == "":
            raw = "Ethiopia is "
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

for strategy, kwargs in [
    ("greedy", {}),
    ("temperature", {"temperature": 0.8}),
    ("temperature", {"temperature": 1.5}),
    ("top_k", {"top_k": 5, "temperature": 1.0}),
    ("top_p", {"top_p": 0.9, "temperature": 1.0}),
]:
    out = model.generate(prompt_ids, max_new_tokens=120, strategy=strategy, **kwargs)
    label = f"{strategy} {kwargs}" if kwargs else strategy
    print(f"\n--- {label} ---\n{decode(out[0])}")

beam_out = beam_search(model, prompt_ids, max_new_tokens=120, beam_width=4)
print(f"\n--- beam_search (width=4) ---\n{decode(beam_out[0])}")


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
