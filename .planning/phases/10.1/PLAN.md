# Phase 10.1: Data generator module

## Objective
Implement the data generation infrastructure for Layer 2. This module is responsible for enumerating RPN programs up to composition depth 2 and computing their numerical signatures at the 6 standard transcendental test points.

## Context
- [docs/LAYER_2_PLAN.md](../../../docs/LAYER_2_PLAN.md) (Step 1)
- [src/eml_transformer/compiler/catalog.py](../../../src/eml_transformer/compiler/catalog.py) (TEST_POINTS, signature_bindings)

## Tasks

### 1. Project Scaffolding
- [ ] Create `src/eml_transformer/layer2/__init__.py`.
- [ ] Create `src/eml_transformer/layer2/dataset.py`.

### 2. Core Datatypes
- [ ] Define `SignatureProgramPair` dataclass:
  - `signature: tuple[complex, ...]`
  - `rpn: str`
- [ ] Define `GeneratorConfig` dataclass:
  - `max_composition_depth: int = 2`
  - `max_args_per_call: int = 2`
  - `exclude_entries: frozenset[str] = frozenset()`
  - `unique_signatures_only: bool = True`
  - `seed: int = 0`

### 3. RPN Enumeration
- [ ] Implement `iter_composition_rpns(registry, max_depth, include_variables)`:
  - **Depth 0**: Yield variables (e.g., `"x"`, `"y"`).
  - **Depth 1**: Yield primitives called with variables (e.g., `"x sin"`, `"x y add"`).
  - **Depth 2**: Yield primitives where at least one argument is a Depth 1 expression.
  - Ensure arity matching against `CalleeSpec.arity`.

### 4. Signature Generation & Deduping
- [ ] Implement `generate_pairs(cfg)`:
  - Build registry from `load_catalog()`.
  - Iterate RPNs from `iter_composition_rpns`.
  - For each RPN:
    - `parse_and_expand` → `compile_tree` → `EMLMachine`.
    - Evaluate at `TEST_POINTS` using `signature_bindings`.
  - **Deduplication**:
    - Compare signatures componentwise (tolerance 10⁻¹⁰).
    - If identical, keep the shorter RPN string.
    - Tiebreak with lexicographic order.

## Verification

### Automated Tests
- [ ] Create `tests/test_layer2_dataset.py`.
- [ ] Test `iter_composition_rpns` for `max_depth=1`.
- [ ] Test `generate_pairs` for small subset of primitives.
- [ ] Test deduplication logic.

### Spot Check (as per LAYER_2_PLAN.md)
- [ ] Run the following check:
  ```bash
  uv run python -c "
  from eml_transformer.layer2.dataset import generate_pairs, GeneratorConfig
  pairs = generate_pairs(GeneratorConfig(max_composition_depth=2))
  print(f'generated {len(pairs)} unique pairs')
  for p in pairs[:10]:
      sig = ', '.join(f'{v.real:.3f}' for v in p.signature[:3])
      print(f'  {p.rpn:30s} → [{sig}, ...]')
  "
  ```
- **Expectation**: ~10³ unique pairs. First items should be trivial (`"x"`, `"y"`) followed by simple calls.
