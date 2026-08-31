import torch
from attention import CausalSelfAttentionKVCache

torch.manual_seed(0)

def test_kv_cache_matches_full_recomputation():
    C, N_HEAD = 16, 4
    model = CausalSelfAttentionKVCache(embed_dim=C, n_head=N_HEAD, max_seq_len=32)
    model.eval()

    seq = torch.randn(1, 6, C)

    with torch.no_grad():
        full_out, _ = model(seq, cache=None)

        cache = None
        incremental_outs = []
        for t in range(seq.shape[1]):
            step_out, cache = model(seq[:, t:t + 1, :], cache=cache)
            incremental_outs.append(step_out)
        incremental_out = torch.cat(incremental_outs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-5)


if __name__ == "__main__":
    test_kv_cache_matches_full_recomputation()
    print("Bonus (KV-cache): all tests passed.")
