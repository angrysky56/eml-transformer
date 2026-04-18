# Project: eml-transformer

## Vision
A compiled mathematical substrate for transformer language models. Converts EML (exp-minus-log) expression trees into analytically-constructed transformer weights that compute elementary functions exactly.

## Core Thesis
A transformer's native architecture (attention + FFN + residual stream) can analytically express any fixed-schedule program. By using the EML operator as the FFN activation, we can compile complex mathematical functions directly into transformer weights, enabling verifiable, machine-precision computation within a neural architecture.

## Milestones

### 1. Layer 1: Analytic Compiler (Completed)
- **Goal:** Build the PyTorch infrastructure to compile RPN expression strings into functional transformer weights.
- **Outcome:** 27 elementary functions (exp, ln, sin, cos, etc.) verified to within 10⁻¹² of reference values.
- **Verification:** `uv run python -m eml_transformer.compiler.verify` → 27/27 passed.

### 2. Layer 2: Learned Program Generator (Active)
- **Goal:** Train a small decoder to predict short RPN-with-calls programs that match a target numerical signature.
- **Context:** [docs/LAYER_2_PLAN.md](../docs/LAYER_2_PLAN.md)
- **Success Criteria:** Signature-match ≥ 30% on held-out depth-2 compositions.

## Technical Stack
- **Runtime:** Python 3.12+ (uv)
- **Framework:** PyTorch (analytically constructed modules)
- **Algebra:** EML (exp-minus-log) Sheffer operator
- **Catalog:** [angrysky56/eml-mcp](https://github.com/angrysky56/eml-mcp)
