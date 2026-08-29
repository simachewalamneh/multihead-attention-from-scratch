#!/usr/bin/env bash
# Rebuilds the repo's git history with ONE COMMIT PER FILE, in dependency
# order, then pushes to GitHub.
#
# Usage:
#   1. Edit REMOTE_URL below to your new (empty) GitHub repo's URL.
#   2. cd into the extracted mha_project directory.
#   3. bash init_and_push.sh
#
# Safe to re-run: it wipes any existing .git in the current directory
# before starting, so it always produces a clean, deterministic history.

set -e

REMOTE_URL="https://github.com/simachewalamneh/multihead-attention-from-scratch.git"
GIT_NAME="Simachew Alamneh"
GIT_EMAIL="simachewalamneh@bongau.edu.et"

rm -rf .git
git init -q
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"

commit_file () {
  local file="$1"
  local msg="$2"
  git add "$file"
  git commit -q -m "$msg"
  echo "  committed: $file"
}

echo "Building commit history (one file per commit)..."

commit_file ".gitignore" \
  "Add .gitignore"

commit_file "README.md" \
  "Add README: project overview, structure, and design-choice notes"

commit_file "attention/single_head.py" \
  "Part 1: SingleHeadAttention — Q/K/V projections, scaled dot-product attention, softmax"

commit_file "attention/causal_single_head.py" \
  "Part 2: CausalSingleHeadAttention — lower-triangular mask applied pre-softmax"

commit_file "attention/multi_head.py" \
  "Part 3: CausalSelfAttention — multi-head split/merge, batched across heads"

commit_file "attention/kv_cache.py" \
  "Bonus: CausalSelfAttentionKVCache — incremental decoding via a growing K/V cache"

commit_file "attention/tied_qkv.py" \
  "Bonus: CausalSelfAttentionTiedQKV — fused Q/K/V projection, numerically equivalent"

commit_file "attention/__init__.py" \
  "Wire up the attention package: re-export all five attention classes"

commit_file "tests/__init__.py" \
  "Add tests package"

commit_file "tests/test_part1_single_head.py" \
  "Test Part 1: shape, row-sums-to-1, embed_dim generalization, unscaled-score saturation"

commit_file "tests/test_part2_causal_mask.py" \
  "Test Part 2: causal invariance under appended tokens, row sums, first-token masking"

commit_file "tests/test_part3_multi_head.py" \
  "Test Part 3: divisibility assertion, output shape, causality, per-head independence"

commit_file "tests/test_bonus_kv_cache.py" \
  "Test Bonus: KV-cache output matches full recomputation exactly"

commit_file "tests/test_bonus_weight_tying.py" \
  "Test Bonus: tied-QKV output matches separate-projection output exactly"

commit_file "run_tests.py" \
  "Add run_tests.py as a single entry point for the whole test suite"

commit_file "model/block.py" \
  "Extension: TransformerBlock — attention + MLP sub-layers, pre-norm, residual connections"

commit_file "model/sampling.py" \
  "Extension: decoding strategies — greedy, temperature, top-k, top-p (nucleus)"

commit_file "model/gpt.py" \
  "Extension: GPTLite decoder-only Transformer, generate() with KV-cache toggle, beam_search()"

commit_file "model/__init__.py" \
  "Wire up the model package: re-export GPTLite and beam_search"

commit_file "tests/test_model_generation.py" \
  "Test extension: GPTLite forward pass, weight tying, generation determinism, cached-vs-uncached equivalence, beam search"

commit_file "generate_text.py" \
  "Extension: end-to-end demo — train on a toy corpus, generate with all 5 strategies, benchmark KV-cache speed"

commit_file "demonstrate_scrambling_bug.py" \
  "Add worked example of the Part 3 reshape/transpose scrambling bug"

echo ""
echo "History built: $(git log --oneline | wc -l) commits."
git log --oneline

echo ""
echo "Pushing to $REMOTE_URL ..."
git branch -M main
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"
git push -u origin main --force

echo ""
echo "Done."
