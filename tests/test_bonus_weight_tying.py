import torch
from attention import CausalSelfAttention, CausalSelfAttentionTiedQKV

torch.manual_seed(0)


def test_tied_qkv_numerically_matches_separate_projections():
    C, N_HEAD = 16, 4
    separate = CausalSelfAttention(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    tied = CausalSelfAttentionTiedQKV(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    tied.load_from_separate(separate)

    separate.eval()
    tied.eval()

    x = torch.randn(1, 6, C)
    with torch.no_grad():
        out_sep = separate(x)
        out_tied = tied(x)

    assert torch.allclose(out_sep, out_tied, atol=1e-5)


if __name__ == "__main__":
    test_tied_qkv_numerically_matches_separate_projections()
    print("Bonus (weight tying): all tests passed.")
