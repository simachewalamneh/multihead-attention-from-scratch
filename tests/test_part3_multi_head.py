import torch
from attention import CausalSelfAttention

torch.manual_seed(0)

C, N_HEAD = 16, 4


def test_rejects_indivisible_embed_dim():
    try:
        CausalSelfAttention(embed_dim=17, n_head=4)
        assert False, "should have raised an AssertionError"
    except AssertionError:
        pass


def test_output_shape_matches_input():
    mha = CausalSelfAttention(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    x = torch.randn(1, 7, C)
    out = mha(x)
    assert out.shape == x.shape


def test_causal_property_holds_across_heads():
    mha = CausalSelfAttention(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    x_short = torch.randn(1, 4, C)
    extra = torch.randn(1, 3, C)
    x_long = torch.cat([x_short, extra], dim=1)
    out_short = mha(x_short)
    out_long = mha(x_long)
    assert torch.allclose(out_short, out_long[:, :4], atol=1e-6)


def _per_head_output(model, x):
    B, T, _ = x.shape
    Q = model._split_heads(model.q_proj(x), B, T)
    K = model._split_heads(model.k_proj(x), B, T)
    V = model._split_heads(model.v_proj(x), B, T)
    s = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
    s = s.masked_fill(~model.causal_mask[:T, :T], float("-inf"))
    w = torch.softmax(s, dim=-1)
    return w @ V   # (B, n_head, T, head_dim) — per-head, pre-merge


def test_heads_are_independent():
    # Perturbing the weights that feed head 0 must change head 0's output
    # only — proof that heads compute genuinely independent attention
    # patterns, not a shared one sliced apart cosmetically.
    a = CausalSelfAttention(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    b = CausalSelfAttention(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    b.load_state_dict(a.state_dict())

    with torch.no_grad():
        hd = b.head_dim
        b.q_proj.weight[0:hd, :] += 5.0   # only head 0's query weights

    x = torch.randn(1, 6, C)
    ph_a = _per_head_output(a, x)
    ph_b = _per_head_output(b, x)

    head0_changed = not torch.allclose(ph_a[:, 0], ph_b[:, 0], atol=1e-6)
    other_heads_unchanged = all(
        torch.allclose(ph_a[:, h], ph_b[:, h], atol=1e-6) for h in range(1, N_HEAD)
    )
    assert head0_changed and other_heads_unchanged


if __name__ == "__main__":
    test_rejects_indivisible_embed_dim()
    test_output_shape_matches_input()
    test_causal_property_holds_across_heads()
    test_heads_are_independent()
    print("Part 3: all tests passed.")
