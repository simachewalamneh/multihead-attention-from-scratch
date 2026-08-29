import torch
from attention import CausalSingleHeadAttention

torch.manual_seed(0)


def _make_model(C=16, max_seq_len=32):
    return CausalSingleHeadAttention(embed_dim=C, max_seq_len=max_seq_len)


def test_earlier_tokens_unaffected_by_appended_tokens():
    C = 16
    attn = _make_model(C)
    x_short = torch.randn(1, 4, C)
    extra = torch.randn(1, 3, C)
    x_long = torch.cat([x_short, extra], dim=1)

    out_short = attn(x_short)
    out_long = attn(x_long)
    assert torch.allclose(out_short, out_long[:, :4], atol=1e-6), (
        "Causal violation: earlier tokens were influenced by later tokens!"
    )


def test_weights_sum_to_one_including_first_token():
    C = 16
    attn = _make_model(C)
    x = torch.randn(1, 7, C)
    Q, K, V = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    scores = (Q @ K.transpose(-2, -1)) / (attn.head_dim ** 0.5)
    mask = attn.causal_mask[:7, :7]
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 7), atol=1e-6)


def test_first_token_attends_only_to_itself():
    C = 16
    attn = _make_model(C)
    x = torch.randn(1, 7, C)
    Q, K, V = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    scores = (Q @ K.transpose(-2, -1)) / (attn.head_dim ** 0.5)
    mask = attn.causal_mask[:7, :7]
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    assert torch.allclose(weights[0, 0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(weights[0, 0, 1:], torch.zeros(6), atol=1e-6)


if __name__ == "__main__":
    test_earlier_tokens_unaffected_by_appended_tokens()
    test_weights_sum_to_one_including_first_token()
    test_first_token_attends_only_to_itself()
    print("Part 2: all tests passed.")
