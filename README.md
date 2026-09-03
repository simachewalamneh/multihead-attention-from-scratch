# Multi-Head Attention from Scratch

Project"Core AI Architecture: Multi-Head
Attention from Scratch".

Implemented using only `nn.Linear` and raw tensor operations
(`matmul`, `softmax`, `tril`, `view`/`transpose`/`reshape`). No
`torch.nn.MultiheadAttention`, no `F.scaled_dot_product_attention`, no
`einops`.

## Structure

```
attention/                      The tasks itself — one module per part
  single_head.py                 Part 1: SingleHeadAttention
  causal_single_head.py          Part 2: CausalSingleHeadAttention
  multi_head.py                  Part 3: CausalSelfAttention
  kv_cache.py                    Bonus:  CausalSelfAttentionKVCache
  tied_qkv.py                    Bonus:  CausalSelfAttentionTiedQKV
  __init__.py                    re-exports all five classes

data/                            Datasets for training and generation
 amharic_ethiopia.txt           Amharic text corpus
 tinyshakespeare.txt            Tiny Shakespeare text corpus
model/                           Extension: attention -> actual text generation
  block.py                       TransformerBlock (attention + MLP + residuals + pre-norm)
  gpt.py                         GPTLite (embeddings + N blocks + vocab head) and beam_search
  sampling.py                    greedy / temperature / top-k / top-p decoding strategies

tests/                           One test module per part/bonus/model component
run_tests.py                     Single entry point: python run_tests.py
generate_text.py                 Trains GPTLite on a toy corpus, generates with every
                                  strategy from the same prompt, benchmarks KV-cache speed
demonstrate_scrambling_bug.py    Isolated demo of the reshape/transpose bug
                                   
```

## `attention/` — the core project

| Class | Task part | What it adds |
|---|---|---|
| `SingleHeadAttention` | Part 1 | Q/K/V projections, scaled dot-product attention, softmax |
| `CausalSingleHeadAttention` | Part 2 | Lower-triangular causal mask applied to scores before softmax |
| `CausalSelfAttention` | Part 3 | Splits into `n_head` heads, runs them as one batched op, concatenates, final projection |
| `CausalSelfAttentionKVCache` | Extra | Incremental (one-token-at-a-time) decoding using a growing K/V cache |
| `CausalSelfAttentionTiedQKV` | Extra | Q/K/V collapsed into a single `Linear(C, 3C)`, numerically identical to the separate-projection version |

## `model/` — extension: attention embedded in an actual generator

Attention alone doesn't generate text; it's one sub-layer in a
Transformer block, which is itself one layer in a stack, which produces
logits that still need to be turned into a chosen token. `model/` closes
that loop:

- **`GPTLite`**: token + positional embeddings → `n_layer` `TransformerBlock`s
  (each using `CausalSelfAttentionKVCache` as its attention sub-layer) →
  final LayerNorm → a vocabulary projection that's *weight-tied* to the
  input token embedding (a second, different use of "weight tying" from
  the Q/K/V tying in `attention/tied_qkv.py` — worth distinguishing the two
  if asked).
- **Five decoding perspectives**, all reading the exact same next-token
  logits differently:
  - `greedy` — always the single most likely token; deterministic, prone to repetition loops.
  - `temperature` — sample from a sharpened/flattened softmax; `temperature -> 0` recovers greedy.
  - `top_k` — sample only from the k highest-probability tokens (fixed-size candidate set).
  - `top_p` (nucleus) — sample from the smallest set of tokens whose cumulative probability exceeds p (candidate set size adapts to how confident the model is at each step).
  - `beam_search` — tracks the top-`beam_width` cumulative-log-probability *sequences* at every step, not just the top token; a different, sequence-level notion of "most likely," and still deterministic like greedy.
- **KV-cache benchmark**: `generate_text.py` runs the identical greedy
  generation with `use_cache=True` vs `False` and times both, so the
  O(T²) → O(T) argument for the KV-cache bonus is demonstrated, not just
  asserted. `tests/test_model_generation.py` also proves the two paths
  produce byte-identical output — caching is a speed optimization, never
  a change in what gets computed.

Run the whole thing end to end:

```bash
python generate_text.py
```

## Design choices 

- **Q/K/V use three separate `nn.Linear` layers**, not one — each learns a
  different role from the same input embedding. The bonus weight-tying
  version shows this can be refactored into one stacked `Linear(C, 3C)`
  without changing the math, which is a different thing from *architecturally*
  tying them.
- **Scaling by `sqrt(head_dim)`**: dot products of two random vectors grow in
  variance with dimension, so scores get large in high dimensions and push
  softmax into a saturated, low-gradient regime. Dividing by `sqrt(head_dim)`
  renormalizes that variance back to ~1 regardless of dimension.
- **Causal mask applied to scores, before softmax, with `-inf`** — not to the
  post-softmax weights. Masking after softmax would break the row-sum-to-1
  guarantee and require a second renormalization; `-inf` before softmax
  gives exactly 0% weight in one step, and `exp(-inf) = 0` exactly, not
  approximately.
- **The head-split reshape is `view(B,T,C) -> view(B,T,n_head,head_dim) ->
  transpose(1,2)`**, never a direct `view(B,n_head,T,head_dim)`. The latter
  reinterprets memory in the wrong order and silently mixes values across
  token boundaries — same output *shape*, wrong numbers. See
  `demonstrate_scrambling_bug.py` for a worked example with traceable
  per-token values.
- **`.contiguous()` after every `.transpose()`** that's followed by a
  `.view()` — transpose returns a non-contiguous view (permuted strides,
  same memory), and `.view()` requires contiguous memory to reinterpret
  correctly.

## Running

```bash
pip install torch
python run_tests.py                    # all 20 tests across attention/ and model/
python demonstrate_scrambling_bug.py    # the Part 3 reshape-bug worked example
python generate_text.py                 # train + generate with all 5 strategies + benchmark
```

`tests/` maps directly onto the projects's own bullet points: attention
weights summing to exactly 1.0, generalization across embedding
dimension, causal invariance under appended tokens, the
divisibility-assertion requirement, per-head independence (perturbing one
head's weights leaves the others provably unchanged), KV-cache vs.
full-recompute equivalence (at both the attention-layer level and the
full generation level), tied-vs-separate weight equivalence, and the
generation pipeline itself (shape, determinism, prompt preservation,
decoding-strategy bounds).
# References

1. **Vaswani, A., et al. (2017). *Attention Is All You Need*.**
   Foundational reference for the Transformer architecture, scaled dot-product attention, and multi-head attention.
   [Paper](https://arxiv.org/abs/1706.03762)

2. **Transformer Core Concepts and Implementation Material.**
   Reference material for understanding Transformer fundamentals, attention, tensor operations, and implementation concepts.
   [Reference Material](https://docs.google.com/document/d/1qYFnO9GJ0nJ51TXD36huJSWsc-EzwV3tdSiNeSfwcQY/edit?tab=t.0)
