# eml-transformer

Research prototype for an **effort-gated compiled substrate** inside transformer language models, using EML (exp-minus-log) trees as a verifiable mathematical IR.

**Status:** Phase 2 Complete. Self-Aware architecture implemented and validated on next-token prediction task.

## The idea in one paragraph

A transformer forward pass could, in principle, allocate more computation to tokens that "deserve" it. Existing adaptive-computation methods (Mixture-of-Recursions, Mixture-of-Depths) learn this allocation via auxiliary losses with no ground truth signal. This project asks: *what if ground truth existed?* EML-compiled math expressions have an exact per-node tree depth — a provably correct "computational complexity" label. Train a router with that supervision on math-heavy data and you get a learned effort evaluator grounded in something measurable, which can then generalize to arbitrary text.

## Four stacked concepts

The long-term architecture stacks four ideas. The Effort Evaluator is the keystone — the other three consume its signal:

1.  **Effort Evaluator** (Phase 1) — per-token scalar effort score, supervised by EML compilation depth.
2.  **Self-Aware layer** (Phase 2) — FFN with `W = W_fixed + effort × ΔW_learnable`. At effort=0 the layer computes a proven EML identity exactly; as effort rises the learned delta activates.
3.  **Spock attention** — attention temperature `τ = 1 − effort`. Low effort stays soft and probabilistic; high effort goes hard and deterministic.
4.  **Maybe Math** — routing specialization: when effort is high *and* content is math-like, dispatch to a fully compiled EML circuit instead of a general attention layer.

## What's in this repository

- `src/eml_transformer/data/` — EML tree generation, tokenization, and depth labeling.
- `src/eml_transformer/models/` — Tiny decoder transformer, the **EffortHead** regression module, and the **Self-Aware** decoder.
- `src/eml_transformer/training/` — Training loops for both Phase 1 (Regression) and Phase 2 (Language Modeling), metrics calculation, and statistical baselines.
- `src/eml_transformer/cli.py` — Unified entry point for data inspection, baseline fitting, and multi-phase model training.

## Results: Summary

### Phase 1 (Effort Evaluator)
The Effort Evaluator achieves **100.00% accuracy** on EML depth prediction, significantly outperforming token-class baselines (63.76%). This confirms that hierarchical complexity is fully learnable from linearized RPN context.

### Phase 2 (Self-Aware Decoder)
We evaluated the Self-Aware (SA) architecture against a Non-Self-Aware (NSA) baseline of identical hidden dimension on the next-token prediction task:

| Model | Accuracy (%) | Cross Entropy Loss |
| :--- | :--- | :--- |
| **Non-Self-Aware (NSA)** | 56.20% | 1.4881 |
| **Self-Aware (SA)** | **56.43%** | **1.4811** |

The SA model shows a measurable edge in both accuracy and loss, suggesting that the effort-modulated delta-branch effectively utilizes the complexity signal provided by the frozen evaluator.

## Setup

Requires Python 3.12+ and CUDA-capable PyTorch 2.6+.

```bash
# Clone and setup environment
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

## Usage

The project follows a two-phase training protocol.

### Phase 1: Train the Effort Evaluator
Train the router to predict EML depth.
```bash
uv run eml-transformer train --save-path evaluator.pt --epochs 10
```

### Phase 2: Train the Self-Aware Transformer
Integrate the frozen evaluator and train the language model.
```bash
# Train Self-Aware model (default)
uv run eml-transformer train-main --evaluator-path evaluator.pt --save-path sa_model.pt

# Train Non-Self-Aware baseline for comparison
uv run eml-transformer train-main --evaluator-path evaluator.pt --save-path nsa_model.pt --no-self-aware
```

### Evaluation
```bash
uv run eml-transformer eval-main --load-path sa_model.pt --evaluator-path evaluator.pt
```

## Why this may not work (Critical Analysis)

As we move from toy verification to research integration, three primary risks remain:

1.  **Marginal Gains vs. Parameter Count**: The Self-Aware FFN effectively doubles the parameter count for the FFN layers (`W_fixed` + `ΔW`). The 0.23% accuracy gain, while measurable, must be weighed against this overhead. The next logical step is **gated execution**, where the `ΔW` branch is skipped entirely for low-effort tokens to reclaim computational efficiency.
2.  **Effort Saturation**: If the Effort Evaluator is too accurate (100%), the modulation signal might become too "binary" for the delta branch to learn smooth transitions. We may need to introduce temperature or noise to the effort signal during Phase 2 training.
3.  **Generalization to Natural Language**: The current dataset is 100% EML math. While the architecture works on this distribution, the "Holy Grail" is generalizing this effort-signal to natural language where "ground truth depth" is not explicitly defined in the training set.

## Testing

```bash
uv run pytest
```
Verification suite covers tree generation, tokenization, model shapes, and checkpoint serialization.
