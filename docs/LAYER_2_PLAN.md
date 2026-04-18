# Layer 2 implementation plan: signature-to-RPN generator

This is a structured plan for implementing the Layer 2 learned program generator for eml-transformer. It is detailed enough that an IDE-integrated assistant can execute it step by step without needing additional design input, but each step also has explicit stop-and-verify criteria so the assistant should pause for human review between steps rather than racing to the end.

**Reviewer: Ty.** When a step completes, run the verification commands, paste the output back to the planning assistant, and wait for a go-ahead before proceeding to the next step.

---

## Context and goal

Layer 1 is complete: a PyTorch module that compiles any EML RPN string to a transformer whose forward pass computes the encoded function to machine precision. The library exposes 27 callable primitives (`sin`, `ln`, `exp`, etc.) through composition, so `parse_and_expand("x ln sin", registry)` produces a compiled machine for `sin(ln(x))` in three tokens.

Layer 2's job: **given the numerical signature of a target function at the 6 standard transcendental test points, output a short RPN-with-calls program that, when compiled and run on the same test points, produces a matching signature to within 10⁻⁸.**

If this works at all, it demonstrates the factoring thesis: a small learned model can navigate a function algebra where every atom is a verified compiled primitive. The model doesn't approximate sin(x); it learns when to emit the token `sin`.

## Non-goals for this phase

- No natural language input. Inputs are purely numerical signatures.
- No novel constant discovery. Programs use the catalog's existing constants plus `1.0`.
- No training at scale. This is a proof-of-concept on a single GPU, measured in hours.
- No reinforcement learning yet. Pure supervised seq2seq first; verifier-in-the-loop comes after the supervised baseline works.

---

## Step 1 — Data generator module

**File:** `src/eml_transformer/layer2/dataset.py` (new package, create `src/eml_transformer/layer2/__init__.py` too).

**Dependencies to import from existing code:**
- `eml_transformer.compiler.catalog.load_catalog`, `TEST_POINTS`, `signature_bindings`
- `eml_transformer.compiler.composer.build_registry`, `parse_and_expand`, `CalleeSpec`
- `eml_transformer.compiler.machine.EMLMachine`, `compile_tree`

### What it produces

A `SignatureProgramPair` dataclass with two fields:
- `signature: tuple[complex, ...]` — 6 complex values, one per TEST_POINT
- `rpn: str` — the RPN string that generates this signature

### How it generates pairs

Three categories of program, yielded in roughly equal proportions:

**(a) Catalog primitives.** For each catalog entry with ≤ 2 variables, emit the RPN form where the entry is called as a single token with variable names as arguments. E.g. `sin` becomes `"x sin"`, `add` becomes `"x y add"`. Signatures are already stored in the catalog — load them directly, don't recompute.

**(b) Single compositions.** Pick a unary outer (e.g. `sin`) and any inner (unary or a specific argument). Build the RPN string `<inner_expr> <outer>` (where inner_expr might itself be a single VAR, a single CALL, or a nested composition up to some depth). Compile via `parse_and_expand`, run on the 6 signature bindings, record the resulting 6 complex values as the signature.

**(c) Variable identity.** Include the trivial case: signature of `x` itself (just the test points), signature of `y`, etc. Programs `"x"`, `"y"`. These are the base cases the decoder needs to learn.

### Configuration

```python
@dataclass(frozen=True)
class GeneratorConfig:
    max_composition_depth: int = 2     # 1 = single call, 2 = nested once
    max_args_per_call: int = 2          # binary ops included
    exclude_entries: frozenset[str] = frozenset()  # e.g. drop "discovered_*"
    unique_signatures_only: bool = True  # dedupe by signature fingerprint
    seed: int = 0
```

### Dedup rule

Two programs with signatures matching componentwise within 10⁻¹⁰ are considered the same function. Keep the shorter program (by token count); tiebreak by lexicographic order. This prevents the model from learning to predict multiple equally-valid forms for the same target — we want it to pick one canonical form.

### API

```python
def generate_pairs(cfg: GeneratorConfig) -> list[SignatureProgramPair]: ...

def iter_composition_rpns(
    registry: dict[str, CalleeSpec],
    max_depth: int,
    *,
    include_variables: tuple[str, ...] = ("x", "y"),
) -> Iterator[str]:
    """Yield every well-formed RPN-with-calls up to the given composition depth.
    Enumerative, not random; for small depths this is finite and small."""
```

The enumerator is the tricky bit. For `max_depth=1` over unary primitives only, it's every `"<var> <primitive>"` — 2 vars × 12 unary primitives = 24 programs. For `max_depth=2` it grows combinatorially but is still bounded; cap the total output at ~10⁴ pairs.

### Verification

After implementing:

```bash
cd /home/ty/Repositories/ai_workspace/eml-transformer
uv run python -c "
from eml_transformer.layer2.dataset import generate_pairs, GeneratorConfig
pairs = generate_pairs(GeneratorConfig(max_composition_depth=2))
print(f'generated {len(pairs)} unique pairs')
# Spot-check
for p in pairs[:10]:
    sig = ', '.join(f'{v.real:.3f}' for v in p.signature[:3])
    print(f'  {p.rpn:30s} → [{sig}, ...]')
"
```

Expected: several hundred to a few thousand pairs; the first few should include `"x"`, `"y"`, `"x sin"`, `"x cos"`, `"x ln"`, etc. If the count is in the millions, the enumerator is over-producing and needs a cap.

**Stop here and report the pair count + first-10 output to Ty before proceeding.**

---

## Step 2 — Tokenizer for Layer 2

**File:** `src/eml_transformer/layer2/tokenizer.py`

A fixed-vocabulary tokenizer distinct from the one in `data/tokenizer.py` (which is for RPN-over-primitives-only, without call tokens).

### Vocabulary

- `<pad>` = 0
- `<bos>` = 1
- `<eos>` = 2
- `<unk>` = 3  (should never be emitted in well-formed programs; present as a safety valve)
- Variables: `x`, `y`, `z` (three slots; most programs use at most two)
- Literal: `1.0` (the only constant the generator is allowed to emit directly — others come via catalog calls like `e`)
- Operator: `E`
- All 27 catalog names as single tokens

Total vocab size is small — around 35. Use a frozen dict built from the catalog; don't hardcode the 27 names since the catalog can grow.

### API

```python
class Layer2Tokenizer:
    @classmethod
    def from_catalog(cls, catalog: list[CatalogEntry]) -> "Layer2Tokenizer": ...

    def encode(self, rpn: str, *, add_special: bool = True) -> list[int]: ...
    def decode(self, ids: list[int], *, strip_special: bool = True) -> str: ...

    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def bos_id(self) -> int: ...
    @property
    def eos_id(self) -> int: ...
```

`encode("x sin")` should produce `[bos_id, token_id("x"), token_id("sin"), eos_id]`.

### Verification

```python
from eml_transformer.compiler.catalog import load_catalog
from eml_transformer.layer2.tokenizer import Layer2Tokenizer

tok = Layer2Tokenizer.from_catalog(load_catalog())
print(f"vocab_size = {tok.vocab_size}")
ids = tok.encode("x sin")
print(f"'x sin' → {ids} → {tok.decode(ids)!r}")
assert tok.decode(ids) == "x sin"

# Round-trip test on a harder case:
for rpn in ["x", "x y add", "x ln sin", "1.0 x E"]:
    assert tok.decode(tok.encode(rpn)) == rpn, rpn
```

**Stop here and confirm round-trips pass before proceeding.**

---

## Step 3 — PyTorch Dataset and collation

**File:** `src/eml_transformer/layer2/torch_dataset.py`

### Input encoding

The signature is 6 complex values = 12 real numbers. Pass them through as a `(12,)` float tensor. Do *not* normalize yet — save that for later if it's needed. Reason: the signatures span large ranges (sin takes values in [-1, 1], exp takes values up to ~15 at the test points), and we want to see whether the model learns the range distribution or whether normalization is required.

### Target encoding

The RPN token stream, encoded via `Layer2Tokenizer`, with BOS prepended and EOS appended. Labels are the standard LM shift: token `i+1` is the prediction target at position `i`, with `pad_id` for positions past EOS.

### Dataset class

```python
class SignatureProgramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: list[SignatureProgramPair],
        tokenizer: Layer2Tokenizer,
        max_target_length: int = 16,
    ): ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...
```

`__getitem__` returns `{"signature": Tensor(12,), "input_ids": Tensor(L,), "labels": Tensor(L,)}` where `L <= max_target_length`. Programs longer than `max_target_length` are dropped during dataset construction (not truncated — truncation would produce ill-formed RPN).

### Collation

Standard left-padding for decoder inputs, right-padding for labels (with pad_id as the ignore index for CrossEntropyLoss). Return `{"signature", "input_ids", "labels", "attention_mask"}`.

### Verification

Build the dataset from step-1 pairs, instantiate a DataLoader with batch_size=4, iterate one batch, print shapes. All shapes should be consistent across the batch.

**Stop and confirm batch shapes look right.**

---

## Step 4 — Model

**File:** `src/eml_transformer/layer2/model.py`

A small seq2seq: a signature encoder (feedforward; the signature is order-invariant across test points but we can treat it as a fixed 12-dim vector for simplicity) and a causal decoder that attends to the encoder output.

### Architecture

```python
@dataclass(frozen=True)
class Layer2Config:
    vocab_size: int
    signature_dim: int = 12
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2    # signature encoder is small — 12 inputs → d_model
    n_decoder_layers: int = 4
    max_target_length: int = 16
    dropout: float = 0.1


class SignatureProgramModel(nn.Module):
    """Encoder + cross-attending decoder.

    Encoder: MLP 12 → d_model → d_model (+ activation), broadcast to a
    fixed-length context of 1 token (or, if d_model is small enough to
    fit signature as 6 complex-pair tokens, project each complex pair
    to d_model and treat as a 6-token sequence). Start with the single-
    token version; upgrade later if needed.

    Decoder: standard causal transformer with cross-attention to encoder
    output. RoPE on self-attention; simple learned positional embeddings
    on cross-attention are fine at this scale.
    """
```

Reuse `eml_transformer.models.layers` where possible — `FeedForward`, `RotaryEmbedding`, the attention mechanism. Build a new `CrossAttention` layer if one doesn't exist (it probably doesn't, since the old code was decoder-only). Keep the cross-attention simple: Q from decoder stream, K/V from encoder output.

### API

```python
def forward(
    self,
    signature: Tensor,      # (B, 12)
    input_ids: Tensor,       # (B, L)
    attention_mask: Tensor,  # (B, L)
) -> Tensor:  # (B, L, vocab_size) logits
    ...

@torch.no_grad()
def generate(
    self,
    signature: Tensor,      # (B, 12)
    *,
    max_length: int = 16,
    bos_id: int,
    eos_id: int,
    temperature: float = 0.0,  # 0.0 = greedy
) -> list[list[int]]:
    """Autoregressive greedy/sampled generation. Returns a batch of
    token-id lists, each stopping at the first EOS (or max_length)."""
```

### Parameter count target

Should be around 500K-1M params for d_model=128 / 4 decoder layers. If much larger, reduce d_model to 96.

### Verification

```python
import torch
from eml_transformer.layer2.model import SignatureProgramModel, Layer2Config
from eml_transformer.layer2.tokenizer import Layer2Tokenizer
from eml_transformer.compiler.catalog import load_catalog

tok = Layer2Tokenizer.from_catalog(load_catalog())
cfg = Layer2Config(vocab_size=tok.vocab_size)
model = SignatureProgramModel(cfg)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# Smoke test forward
sig = torch.randn(2, 12)
ids = torch.randint(4, tok.vocab_size, (2, 8))
mask = torch.ones_like(ids, dtype=torch.bool)
logits = model(sig, ids, mask)
assert logits.shape == (2, 8, tok.vocab_size), logits.shape

# Smoke test generate
out = model.generate(sig, max_length=10, bos_id=tok.bos_id, eos_id=tok.eos_id)
print("generate output lengths:", [len(o) for o in out])
```

**Stop and report parameter count + smoke test output.**

---

## Step 5 — Supervised training loop

**File:** `src/eml_transformer/layer2/train.py`

Standard teacher-forced cross-entropy training. The point of this step is not to produce a great model — it's to confirm that the architecture can *fit* the training distribution at all. If it can't fit 1000 training pairs, no amount of RL or data scaling will help.

### Training config

```python
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    device: str = "cuda"
    eval_every: int = 1   # epochs
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0
    save_path: str | None = None
```

Dataset split: 80/10/10 train/val/test. Splitting **by program depth** is important — put all `max_depth=2` compositions in the held-out val+test, keep `max_depth=1` in train. This tests whether the model learns composition as an algebra, not just memorizes.

### Metrics

Report per-epoch on val:
- **Teacher-forced CE loss** (standard).
- **Greedy decode exact match** — fraction of val programs where `tokenizer.decode(greedy_generate())` equals `pair.rpn` exactly.
- **Signature match** — compile the generated RPN via `parse_and_expand` + `compile_tree` + `EMLMachine`, run on the 6 test points, compare to target signature with tolerance 10⁻⁸. Handles the case where the model produces a *different but equivalent* program. Failures to compile (malformed RPN, unbalanced stack) count as non-matches but are logged separately.

The signature-match metric is the real one. Token exact-match is secondary and can be much lower than signature-match if the model finds multiple valid forms.

### API

```python
def train(
    pairs: list[SignatureProgramPair],
    tok: Layer2Tokenizer,
    cfg: TrainConfig,
) -> SignatureProgramModel: ...
```

### Verification

Run the training loop with `epochs=5, batch_size=32` on a dataset of ~500 pairs. Expected signals:

- Training loss should decrease monotonically from ~log(vocab_size) ≈ 3.5 to ~0.5-1.0 over 5 epochs.
- Val exact-match should be non-zero by epoch 5 (not necessarily high — even 5% means the model is learning something).
- Val signature-match should be at least as high as exact-match, ideally higher.

If training loss plateaus at the initial value, something is wired wrong (gradients not flowing through cross-attention, label shift off-by-one, etc.).

**Stop and report the per-epoch val metrics before proceeding.** This is the critical decision point: if the model doesn't learn at all, we need to debug rather than continue to RL.

---

## Step 6 — (Conditional on Step 5 succeeding) Verifier-in-the-loop

Only proceed to this step if Step 5's val signature-match is nonzero and appears to improve over training. If Step 5 plateaus at zero, we're debugging, not scaling.

**File:** `src/eml_transformer/layer2/rl.py`

Implement REINFORCE-style fine-tuning using the verifier as reward:
- For each target signature, sample N programs from the current policy.
- Compile each, evaluate on the 6 test points, score as `1.0 - min(err, 1.0)` or similar bounded reward.
- Policy gradient: maximize expected reward with a baseline (batch mean) to reduce variance.
- Regularize with KL to the supervised-pretrained policy to prevent collapse.

Full spec to be written when we get here, because the right shape depends on what Step 5 reveals.

---

## Housekeeping

- All new code follows the project's existing conventions: `>=3.12` syntax, no `typing.Optional` (use `X | None`), dataclasses for config, `from __future__ import annotations` at the top of every file.
- Docstrings on every public function and class.
- No new dependencies. Torch, NumPy, and stdlib only.
- Tests for each new module go in `tests/test_layer2_<module>.py`, matching the existing layout.
- Run `uv run pytest` before declaring any step done. The compiler tests (`tests/test_compiler.py`) must continue to pass untouched — if they break, something in Layer 2 is reaching into Layer 1 incorrectly.

## Paths that should NOT be touched

- `src/eml_transformer/compiler/` — Layer 1 is frozen. Any changes here would invalidate the verifier's 27/27 pass claim. If something in the compiler appears to need modification, stop and ask.
- `src/eml_transformer/models/layers.py` — can be imported from, but new layer types (like `CrossAttention`) belong in `layer2/` to keep the old Phase 2 code isolated.
- `tests/test_compiler.py` — frozen.

## Exit criteria for Layer 2 v0

When all of the following hold, Layer 2 is considered "working proof-of-concept" and we move on to writing it up:

1. `tests/test_layer2_*.py` all pass.
2. `uv run python -m eml_transformer.compiler.verify` still prints `27/27 passed`.
3. On a held-out test set of depth-2 compositions (trained on depth-1 only), the model achieves **signature-match ≥ 30%** at temperature 0.0 generation.
4. Training completes in under 2 hours on a single RTX 3060.

30% might sound low. It isn't. Random baseline is effectively 0% — the output space is ~35 token choices per position over ~8 positions, so ~35⁸ sequences. Any nonzero held-out signature-match means the model has learned to compose catalog entries, which is the whole thesis.
