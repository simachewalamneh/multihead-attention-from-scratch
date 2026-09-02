import torch
from model import GPTLite, beam_search

torch.manual_seed(0)

VOCAB, C, N_HEAD, N_LAYER, MAX_LEN = 12, 16, 2, 2, 32

def _make_model():
    return GPTLite(vocab_size=VOCAB, embed_dim=C, n_head=N_HEAD, n_layer=N_LAYER, max_seq_len=MAX_LEN)

def test_forward_output_shape():
    model = _make_model()
    idx = torch.randint(0, VOCAB, (2, 5))
    logits, cache = model(idx, cache_list=None)
    assert logits.shape == (2, 5, VOCAB)
    assert len(cache) == N_LAYER

def test_weight_tying_shares_storage():
    # head.weight and token_embed.weight must be the literal same tensor
    # (not just equal in value) — that's what "tying" means.
    model = _make_model()
    assert model.head.weight is model.token_embed.weight

def test_generate_extends_sequence_by_requested_length():
    model = _make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))
    out = model.generate(prompt, max_new_tokens=10, strategy="greedy")
    assert out.shape == (1, 13)
    assert torch.equal(out[0, :3], prompt[0])   # prompt is preserved verbatim

def test_greedy_generation_is_deterministic():
    model = _make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))
    out1 = model.generate(prompt, max_new_tokens=10, strategy="greedy")
    out2 = model.generate(prompt, max_new_tokens=10, strategy="greedy")
    assert torch.equal(out1, out2)

def test_cached_and_uncached_generation_produce_identical_tokens():
    # Greedy decoding must pick the exact same tokens whether or not the
    # KV-cache path is used — the cache is a speed optimization, not a
    # change in what gets computed. This is the generation-level version
    # of the equivalence already proven at the attention layer in
    # tests/test_bonus_kv_cache.py.
    model = _make_model()
    model.eval()
    prompt = torch.randint(0, VOCAB, (1, 3))
    out_cached = model.generate(prompt, max_new_tokens=8, strategy="greedy", use_cache=True)
    out_uncached = model.generate(prompt, max_new_tokens=8, strategy="greedy", use_cache=False)
    assert torch.equal(out_cached, out_uncached)

def test_beam_search_output_shape_and_prompt_preserved():
    model = _make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))
    out = beam_search(model, prompt, max_new_tokens=6, beam_width=3)
    assert out.shape == (1, 9)
    assert torch.equal(out[0, :3], prompt[0])

def test_top_k_and_top_p_stay_within_vocab_range():
    model = _make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))
    for strategy, kwargs in [
        ("top_k", {"top_k": 3}),
        ("top_p", {"top_p": 0.8}),
        ("temperature", {"temperature": 1.2}),
    ]:
        out = model.generate(prompt, max_new_tokens=10, strategy=strategy, **kwargs)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

if __name__ == "__main__":
    test_forward_output_shape()
    test_weight_tying_shares_storage()
    test_generate_extends_sequence_by_requested_length()
    test_greedy_generation_is_deterministic()
    test_cached_and_uncached_generation_produce_identical_tokens()
    test_beam_search_output_shape_and_prompt_preserved()
    test_top_k_and_top_p_stay_within_vocab_range()
    print("model/: all tests passed.")
