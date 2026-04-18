# Phase 10.5: Supervised Training Loop

This phase implements the supervised training loop for the Layer 2 model. The primary goal is to verify that the model can learn to navigate the EML primitive space and potentially generalize to deeper compositions.

## Objectives
- Implement `src/eml_transformer/layer2/train.py`.
- Implement a compositional generalization split: Train on depth-1, Test on depth-2.
- Implement training loop with CrossEntropy loss.
- Implement validation metrics: Exact Match and Signature Match.
- Save best model checkpoint.

## Requirements

### Training Configuration
- **Epochs**: 50 (higher because the training set is small if using depth-1 only)
- **Batch Size**: 64
- **LR**: 3e-4 (AdamW)
- **Device**: CUDA if available.
- **Eval Every**: 5 epochs.

### Dataset Split (Compositional Generalization)
1. Generate all pairs up to depth 2.
2. Group by call count:
   - **Train**: Programs with 0 or 1 catalog calls (atoms and depth-1 compositions).
   - **Val/Test**: Programs with 2 catalog calls (depth-2 compositions).
3. If the training set is too small, take a random 10% sample of depth-2 for training as well, but track "Zero-Shot Depth-2" metric separately.

### Metrics
- **CE Loss**: Standard cross-entropy.
- **Exact Match**: Fraction of programs where greedy decode matches target RPN string.
- **Signature Match**: Fraction of programs where generated RPN computes the same signature (within 10⁻⁸) as target.
  - Requires: `eml_transformer.compiler` (machine, compiler, catalog).

## Verification Plan

### Automated Tests
1. Create `tests/test_layer2_train.py`.
2. Test metric computation (Exact Match, Signature Match) on dummy data.
3. Test that the training loop runs for 1 epoch on a small mock dataset.

### Manual Verification
Execute a training run and report:
- Initial and final training loss.
- Val Exact Match.
- Val Signature Match (the key metric).

Expected: Training loss should drop significantly. Signature match on held-out depth-2 should be > 0% (any success is a win for compositional generalization).

```bash
uv run python -m eml_transformer.layer2.train --epochs 20 --batch_size 32
```
