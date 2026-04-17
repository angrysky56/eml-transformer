# eml-transformer

Research prototype for an **effort-gated compiled substrate** inside transformer language models, using EML (exp-minus-log) trees as a verifiable mathematical IR.

**Status:** Phase 1 Complete. Effort Evaluator achieves 100% accuracy on EML depth prediction. Phase 2 (Self-Aware) to follow.

## The idea in one paragraph

A transformer forward pass could, in principle, allocate more computation to tokens that "deserve" it. Existing adaptive-computation methods (Mixture-of-Recursions, Mixture-of-Depths) learn this allocation via auxiliary losses with no ground truth signal. This project asks: *what if ground truth existed?* EML-compiled math expressions have an exact per-node tree depth — a provably correct "computational complexity" label. Train a router with that supervision on math-heavy data and you get a learned effort evaluator grounded in something measurable, which can then generalize to arbitrary text.

## Four stacked concepts

The long-term architecture stacks four ideas. The Effort Evaluator is the keystone — the other three consume its signal:

1. **Effort Evaluator** *(this phase)* — per-token scalar effort score, supervised by EML compilation depth.
2. **Self-Aware layer** — FFN with `W = W_fixed + effort × ΔW_learnable`. At effort=0 the layer computes a proven EML identity exactly; as effort rises the learned delta activates.
3. **Spock attention** — attention temperature `τ = 1 − effort`. Low effort stays soft and probabilistic; high effort goes hard and deterministic.
4. **Maybe Math** — routing specialization: when effort is high *and* content is math-like, dispatch to a fully compiled EML circuit instead of a general attention layer.

Only concept 1 is in scope for this repository at present.

## What's in this repository

- `src/eml_transformer/data/` — EML tree generation, tokenization, and depth labeling.
- `src/eml_transformer/models/` — Tiny decoder transformer and the **EffortHead** regression module.
- `src/eml_transformer/training/` — Training loop, metrics calculation, and statistical baselines.
- `src/eml_transformer/cli.py` — Unified entry point for data inspection, baseline fitting, and model training.

## Results: Phase 1 (Effort Evaluator)

The Effort Evaluator is a transformer-based router that predicts the computational complexity (tree depth) of mathematical tokens. In the initial benchmark:
- **Baseline (Token-Class)**: 63.76% Accuracy
- **Effort Evaluator (Phase 1)**: **100.00% Accuracy**

This confirms that hierarchical complexity is fully learnable from linearized RPN context.

## Next Steps: Phase 2 (Self-Aware Layer)

The next phase integrates the Effort Evaluator into a **Self-Aware layer** where FFN weights are modulated by the predicted effort:
`W = W_fixed + effort × ΔW_learnable`

## Setup

Requires Python 3.12+ and CUDA-capable PyTorch 2.11+.

```bash
cd /home/ty/Repositories/ai_workspace/eml-transformer
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra dev
```

## Smoke test

```bash
uv run pytest
```

Should show ~10 passing tests that verify tree generation, tokenization, and depth labeling are internally consistent.

## Why this might not work (honestly)

Three ways the Effort Evaluator hypothesis could fail, each of which we'd rather learn now than later:

1. **Depth is not learnable from RPN context.** The router might need more than local RPN context to predict subtree depth. If so, the toy is too minimal and we'd need to move to expression-level labels rather than per-token labels.
2. **Depth correlates trivially with something cheaper.** E.g., "token is `E` (eml op)" vs "token is a leaf" might explain most variance, in which case we haven't learned effort so much as token class. Evaluation must test generalization across depth distributions.
3. **Training with exact depth labels makes the router brittle.** A router that's overly confident on math-shaped inputs may fail to generalize to ordinary language. This is the MoR paper's concern in reverse.

The scaffold is intentionally small so we can falsify fast.
