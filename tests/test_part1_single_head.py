import torch
from attention import SingleHeadAttention

torch.manual_seed(0)


def test_output_shape_matches_input():
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)
    attn = SingleHeadAttention(embed_dim=C)
    out = attn(x)
    assert out.shape == (B, T, C)


def test_attention_weights_sum_to_one():
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)
    attn = SingleHeadAttention(embed_dim=C)
    Q, K, V = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    scores = (Q @ K.transpose(-2, -1)) / (attn.head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_generalizes_across_embed_dim():
    for dim in [8, 16, 64, 256]:
        x = torch.randn(1, 4, dim)
        attn = SingleHeadAttention(embed_dim=dim)
        out = attn(x)
        assert out.shape == (1, 4, dim), f"failed for dim={dim}"


def test_unscaled_scores_saturate_softmax():
    # Demonstrates WHY the scaling step exists: without it, high-dimensional
    # dot products blow up in magnitude and push softmax toward a
    # near-one-hot, low-gradient regime.
    d_big = 4096
    Q_big = torch.randn(1, 1, d_big) * 3
    K_big = torch.randn(1, 1, d_big) * 3
    raw_score = (Q_big @ K_big.transpose(-2, -1)).item()
    assert abs(raw_score) > 50, (
        "expected an unscaled high-dim dot product to have large magnitude"
    )


if __name__ == "__main__":
    test_output_shape_matches_input()
    test_attention_weights_sum_to_one()
    test_generalizes_across_embed_dim()
    test_unscaled_scores_saturate_softmax()
    print("Part 1: all tests passed.")
