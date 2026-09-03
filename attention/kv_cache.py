import math
import torch

from .multi_head import CausalSelfAttention

class CausalSelfAttentionKVCache(CausalSelfAttention):
 
    def forward(self, x: torch.Tensor, cache: dict | None = None):
      
        B, T_new, C = x.shape

        Q_new = self._split_heads(self.q_proj(x), B, T_new)   # (B, nh, T_new, hd)
        K_new = self._split_heads(self.k_proj(x), B, T_new)   # (B, nh, T_new, hd)
        V_new = self._split_heads(self.v_proj(x), B, T_new)   # (B, nh, T_new, hd)

        if cache is not None:
            #appends the one new K/V vector onto the end of everything already accumulated
            K = torch.cat([cache["K"], K_new], dim=2)   # concat along the T axis
            V = torch.cat([cache["V"], V_new], dim=2)
        else:
            K, V = K_new, V_new

        T_total = K.shape[2] #take the  3rd dimension

        scores = (Q_new @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T_new, T_total)
     
        mask = self.causal_mask[T_total - T_new:T_total, :T_total]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1) #last dimension represents the key/token positions
        head_out = attn_weights @ V                                        # (B, nh, T_new, hd)

        merged = self._merge_heads(head_out, B, T_new)
        out = self.out_proj(merged)

        new_cache = {"K": K, "V": V}
        return out, new_cache
