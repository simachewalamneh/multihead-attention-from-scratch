"""
Part 3's last bullet asks you to identify the EXACT point where token data
would silently get scrambled if you reorder operations incorrectly.

This script proves it by building the WRONG version side by side with the
correct one. Both versions run without errors and produce a (B, T, C)
tensor of the right SHAPE — that's exactly what makes this bug dangerous:
nothing crashes, the numbers are just wrong.

Wrong version:  x.view(B, n_head, T, head_dim)
Correct version: x.view(B, T, n_head, head_dim).transpose(1, 2)
"""
import torch

B, T, n_head, head_dim = 1, 4, 2, 3
C = n_head * head_dim

# A tensor where we can literally read off which token each value came
# from: token t's embedding is filled entirely with the value (t+1)*10.
x = torch.arange(1, T + 1).float().repeat_interleave(C).view(1, T, C) * 1.0
for t in range(T):
    x[0, t] = (t + 1) * 10
print("Input x (each row is one token, values = token_id * 10):\n", x[0])

# --- WRONG: reshape straight to (B, n_head, T, head_dim) -------------------
# This treats the flat (T*C) memory as if head_dim varies fastest across
# n_head THEN T, which is not what the data layout is. It does not error -
# it just quietly reinterprets which numbers belong to which (head, token).
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

print("\nThe exact point of failure: view() reinterprets a flat memory buffer")
print("assuming the LAST dimension varies fastest. (B,T,C) memory is laid out")
print("token-by-token, so view(B,T,nh,hd) correctly keeps each token's C")
print("values together before splitting them into nh chunks. Going straight")
print("to (B,nh,T,hd) instead makes 'view' walk memory in (n_head, T) order,")
print("which cuts across token boundaries -- it silently reads head_dim")
print("values that span PARTS of two different tokens' embeddings once T>1.")
