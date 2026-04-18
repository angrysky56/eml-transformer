# eml-transformer

A compiled mathematical substrate for transformer language models. Converts EML (exp-minus-log) expression trees into analytically-constructed transformer weights that compute elementary functions exactly — no training, verifiable to machine precision.

**Status:** Layer 1 complete. 27 elementary functions compile and verify to within 10⁻¹² of NumPy reference. Layer 2 (learned program generator) in progress.

## The idea

Large language models compute elementary math statistically — they predict the tokens that look like the answer. On common inputs this produces the illusion of knowing; on uncommon inputs it fails unpredictably. An LLM has no internal mechanism that *actually computes* sin(x) the way a calculator does.

This project builds one. Given that every elementary function reduces to repeated application of a single binary operator `eml(x, y) = exp(x) − ln(y)` (Odrzywołek 2026), and that a transformer's native shape (attention routing + pointwise FFN + residual stream) can analytically express any fixed-schedule program (Moran 2026), we can compile the EML expression for any elementary function directly into transformer weights. The result is a transformer that *runs the computation*, not one that guesses its output.

## What's here (Layer 1)

The compiler takes an RPN expression string from the EML catalog and produces a PyTorch `nn.Module` whose forward pass computes that function on a residual-stream machine:

- **Sequence length** = number of RPN tokens
- **Layers** = tree depth + 1
- **Attention** = analytic index-select routing (each EML node reads its two operands from their precomputed positions)
- **FFN** = the EML operator `exp(a) − ln(b)` applied to the attention output
- **Parameters** = zero (all weights are analytically set buffers)

```python
from eml_transformer.compiler import EMLMachine, build_registry, load_catalog, parse_and_expand, compile_tree

# Compile from a library of 27 verified primitives.
registry = build_registry(load_catalog())

# "x ln sin" → three tokens, machine-epsilon correct sin(ln(x)).
tree = parse_and_expand("x ln sin", registry)
machine = EMLMachine(compile_tree(tree))
machine({"x": 2.0})  # → complex(0.638961276313635 + 0j)
# math.sin(math.log(2.0)) = 0.638961276313635  — bit-identical.
```

### Verification

Every entry in the catalog verifies against its stored six-point transcendental signature to within 10⁻¹²:

```bash
uv run python -m eml_transformer.compiler.verify
# catalog verification: 27/27 passed, 0 failed, 0 skipped
```

Maximum error across the catalog is 1.66×10⁻¹² (on `tan`, a 97-position 24-layer machine). Half the entries are bit-exact.

## What's in the catalog

27 compiled primitives, ranging from `exp` (2 layers, 3 positions, K=3) to `tan` (24 layers, 97 positions, K=97):

- Unary: `exp`, `ln`, `negate`, `reciprocal`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `exp_exp`, `ln_ln`
- Binary: `add`, `subtract`, `multiply`, `divide`
- Nullary: `e`, `zero`
- Plus 9 auto-discovered compositions

The catalog lives in the companion project [`eml-mcp`](https://github.com/angrysky56/eml-mcp) and is loaded as read-only data.

## Directory layout

- `src/eml_transformer/compiler/` — Layer 1. RPN parser, tree composer, compiled machine, verifier.
- `src/eml_transformer/data/`, `models/`, `training/` — earlier work on effort-gated transformers; see `docs/PHASE2_NULL.md` for why those experiments were superseded.
- `tests/test_compiler.py` — 19 tests covering RPN parsing, compilation, composition, and verification.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

## Usage

### Verify the catalog

```bash
uv run python -m eml_transformer.compiler.verify            # summary only
uv run python -m eml_transformer.compiler.verify --show-all  # full table
uv run python -m eml_transformer.compiler.verify --only sin cos tan
```

### Compile and evaluate a function

```python
from eml_transformer.compiler import EMLMachine

# Direct from RPN (no library calls)
exp_machine = EMLMachine.from_rpn("x 1.0 E")
exp_machine({"x": 0.5})  # → exp(0.5)

# Via the library
from eml_transformer.compiler import build_registry, load_catalog, parse_and_expand, compile_tree
registry = build_registry(load_catalog())
tree = parse_and_expand("x sin", registry)  # sin called as one token
sin_machine = EMLMachine(compile_tree(tree))
sin_machine({"x": 0.5})  # → sin(0.5) to machine epsilon
```

### Run tests

```bash
uv run pytest tests/test_compiler.py -v
```

## What this is not

This is not a general-purpose symbolic math system. The catalog covers elementary functions — polynomials via compositions, exp/log, trig, hyperbolic, and their combinations. It does not cover integration, special functions (Bessel, gamma), linear algebra, or any algorithm requiring unbounded iteration. Extending the substrate to those domains is a separate compilation effort per domain.

This is also not a production inference accelerator. A compiled `tan` machine runs 24 transformer layers to produce one float. NumPy's `tan` runs in nanoseconds. The point is not speed — it's that the transformer's forward pass *is* the computation, available as a callable primitive for a learned agent to use.

## What comes next (Layer 2)

A small learned decoder that, given a description of a target function (either numerical samples or a natural-language reference), emits a short RPN program that the Layer 1 machine verifies to machine precision. Uses the verifier as an oracle reward signal. See `docs/LAYER_2_PLAN.md` for the implementation plan.

## References

- Odrzywołek, A. (2026). *EML as a universal minimal operator for elementary functions.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
- Moran, S. (2026). *I built a tiny computer inside a transformer.* [Towards Data Science](https://towardsdatascience.com/i-built-a-tiny-computer-inside-a-transformer/) · [transformer-vm code](https://github.com/Percepta-Core/transformer-vm).
- Companion project: [`angrysky56/eml-mcp`](https://github.com/angrysky56/eml-mcp) — the EML catalog, bootstrapping search, and symbolic verifier.

## License

MIT.
