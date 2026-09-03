import torch

B, T, n_head, head_dim = 1, 4, 2, 3
C = n_head * head_dim

x = torch.arange(1, T + 1).float().repeat_interleave(C).view(1, T, C) * 1.0
for t in range(T):
    x[0, t] = (t + 1) * 10
print("Input x (each row is one token, values = token_id * 10):\n", x[0])

# --- WRONG: reshape straight to (B, n_head, T, head_dim) -------------------

wrong = x.view(B, n_head, T, head_dim)
print("\nWRONG reshape -> (B, n_head, T, head_dim):")
print(wrong[0])
print("Notice head 0's 'tokens' mix values that don't correspond to any")
print("single real token's embedding — token identity has been scrambled.")

# --- CORRECT: view then transpose -------------------------------------------
correct = x.view(B, T, n_head, head_dim).transpose(1, 2)
print("\nCORRECT view(B,T,nh,hd).transpose(1,2) -> (B, n_head, T, head_dim):")
print(correct[0])
print("Every row is still traceable to exactly one original token's values,")
print("just sliced into its head_dim chunk for this head.")