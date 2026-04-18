# Phase 2 null result: effort-gated FFN modulation

**TL;DR:** Effort-modulated FFN variants (Delta, FiLM) did not outperform an unmodulated baseline at matched parameter count on EML language modeling. The project pivoted from effort-gated transformers to compiled EML substrates (Layer 1, see main README).

## What was tried

A frozen Effort Evaluator (Phase 1, 99.8% depth-prediction accuracy on held-out depth-7 trees) was used to produce a per-token effort scalar that modulated the FFN of a downstream "main decoder" on next-token prediction over EML RPN sequences. Three modulation variants were compared:

- **Vanilla** — no modulation, standard GELU FFN (control).
- **Delta** — `fixed(x) + effort · delta(x)`, two parallel FFNs scaled by effort.
- **FiLM** — per-channel γ/β generated from effort, modulating FFN hidden activations before GELU (Perez et al. 2018).

## Parameter-matched depth-7 results (single seed, 1000 eval samples)

| Mode    | Params | LM accuracy |
|---------|--------|-------------|
| Vanilla (expansion=8) | ~1.31M | 54.40% |
| FiLM (expansion=8)    | ~1.33M | 54.49% |
| Delta (expansion=4)   | ~1.31M | 54.60% |

## Why this is a null

Standard error on 1000-sample accuracy at p≈0.5 is approximately 1.58%. The spread between the three modes is 0.20% — roughly 1/8 the noise floor. A three-seed confirmation would be the proper sanity check but was judged not worth the compute given the spread.

The held-out depth-extrapolation test (train depth-5, eval depth-7) was more informative: modulated variants *underperformed* vanilla at OOD depth (50-51% vs 52.75%). This suggests effort modulation *hurts* generalization when the signal range shifts — the decoder receives "maximum effort" signals for trees deeper than it saw during training, and its modulation weights don't know how to respond.

## What was kept

- `evaluator_v7.pt` — the trained depth evaluator. 99.81% total accuracy, 94.97% on depth-7 tokens. Still a genuinely novel artifact — a transformer supervised by provably-correct symbolic labels.
- Fair-comparison infrastructure: `compare-modes` subcommand, parameter-matching helpers, held-out depth evaluation.

## Why the project pivoted rather than iterated

Effort modulation on LM-over-RPN was solving the wrong problem for the original vision. The thesis was "transformer that knows math, not just guesses it" — modulation still guesses, it just guesses with an auxiliary signal. The compiled substrate (Layer 1) actually computes, which is the thing the original concept demanded.

## References

Published work in the same space that covers the modulation direction better than this project would have:

- Bae et al. (2025). *Mixture-of-Recursions: learning dynamic recursive depths for adaptive token-level computation.* [arXiv:2507.10524](https://arxiv.org/abs/2507.10524).
- Wang et al. (2025). *Hierarchical Reasoning Model.* [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).
- Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* [arXiv:1709.07871](https://arxiv.org/abs/1709.07871).
