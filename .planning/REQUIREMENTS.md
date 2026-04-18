# Milestone: Layer 2 — Learned Program Generator

## Goal
Given the numerical signature of a target function at 6 standard transcendental test points, output a short RPN-with-calls program that, when compiled and run on the same test points, produces a matching signature to within 10⁻⁸.

## Success Criteria
- [ ] Model achieves **signature-match ≥ 30%** on a held-out test set of depth-2 compositions (trained on depth-1 only).
- [ ] Training completes in under 2 hours on a single RTX 3060.
- [ ] All tests in `tests/test_layer2_*.py` pass.
- [ ] No regressions in Layer 1 (`verify` still passes 27/27).

## Core Requirements

### 1. Data Generation
- Enumerate programs up to composition depth 2.
- Support catalog primitives (arity ≤ 2), single compositions, and variable identity.
- Deduplicate by signature fingerprint (tolerance 10⁻¹⁰), keeping the shortest program.
- Input: 6 complex test points → Output: RPN string.

### 2. Tokenization
- Vocabulary including special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`), variables (`x`, `y`, `z`), literal `1.0`, operator `E`, and all 27 catalog names.
- Round-trip stability for all RPN programs in the dataset.

### 3. Model Architecture
- Small encoder-decoder transformer (0.5M - 1M params).
- Signature Encoder: MLP projecting 12-dim real vector (6 complex points) to `d_model`.
- Program Decoder: Causal transformer with cross-attention to the encoder output.
- Reuse Layer 1 components (FFN, Rotary, Attention) where possible.

### 4. Training
- Supervised teacher-forcing on depth-1 compositions.
- Evaluation on depth-2 compositions to test algebraic generalization.
- Reward metric: signature match error < 10⁻⁸.

### 5. (Optional) RL Fine-tuning
- REINFORCE-style optimization using the Layer 1 verifier as a reward signal if SL baseline is promising.
