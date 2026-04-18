# Phase 10.4: Model Architecture

This phase implements the Layer 2 model architecture: a small seq2seq transformer with a signature encoder and a causal decoder that attends to the encoder output.

## Objectives
- Implement `src/eml_transformer/layer2/model.py`.
- Implement `SignatureEncoder`: Projects 12-dim signature to `d_model`.
- Implement `SignatureProgramModel`: Encoder + Cross-attending Decoder.
- Reuse existing `eml_transformer.models.layers` components (FFN, RoPE, Attention).
- Verify parameter count (Target: 0.5M - 1.0M).
- Verify forward pass and autoregressive generation.

## Requirements

### Architecture Details
- **d_model**: 128 (default)
- **n_heads**: 4
- **n_decoder_layers**: 4
- **max_target_length**: 16
- **Encoder**: 
  - Version 1: Simple MLP (12 -> d_model -> d_model) broadcast to 1 context token.
- **Decoder**:
  - Causal self-attention (with RoPE).
  - Cross-attention to encoder output.
  - LayerNorm and Residual connections.

### API: `SignatureProgramModel`
```python
class SignatureProgramModel(nn.Module):
    def forward(self, signature: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...
    @torch.no_grad()
    def generate(self, signature: Tensor, max_length: int = 16, bos_id: int = 1, eos_id: int = 2) -> list[list[int]]: ...
```

## Verification Plan

### Automated Tests
1. Create `tests/test_layer2_model.py`.
2. Verify parameter count is within the target range.
3. Test `forward` pass with batch shapes.
4. Test `generate` produces valid-looking token sequences.
5. Verify gradients flow through both self and cross attention.

### Manual Spot Check
Run the verification script from `LAYER_2_PLAN.md` to report parameter count and generation output.

```python
import torch
from eml_transformer.layer2.model import SignatureProgramModel, Layer2Config
from eml_transformer.layer2.tokenizer import Layer2Tokenizer
from eml_transformer.compiler.catalog import load_catalog

tok = Layer2Tokenizer.from_catalog(load_catalog())
cfg = Layer2Config(vocab_size=tok.vocab_size)
model = SignatureProgramModel(cfg)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

sig = torch.randn(2, 12)
out = model.generate(sig, max_length=10, bos_id=tok.bos_id, eos_id=tok.eos_id)
print("generate output lengths:", [len(o) for o in out])
```
