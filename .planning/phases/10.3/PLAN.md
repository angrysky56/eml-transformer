# Phase 10.3: PyTorch Dataset & Collation

This phase bridges the raw `SignatureProgramPair` objects and the `Layer2Tokenizer` into a `torch.utils.data.Dataset` suitable for training the Layer 2 model.

## Objectives
- Implement `src/eml_transformer/layer2/torch_dataset.py`.
- Implement `SignatureProgramDataset` for mapping pairs to tensors.
- Implement collation logic to handle batching, padding, and labels.
- Verify batch consistency and tensor shapes.

## Requirements

### Input Encoding
- Convert 6 complex values (the signature) into a 12-dimensional real vector (concatenating real and imaginary parts: `[r1, i1, r2, i2, ...]`).
- Output: `(12,)` float32 tensor.

### Target Encoding
- Use `Layer2Tokenizer` to encode RPN strings.
- Prepend `<bos>`, append `<eos>`.
- Cap sequence length at `max_target_length` (default 16).
- Drop programs that exceed the cap (don't truncate).

### Collation
- Standard left-padding for `input_ids` (so the last token is always the one before EOS or EOS itself).
- Right-padding for `labels`.
- `labels` should be shifted by 1 relative to `input_ids`.
- Use `pad_id` as the ignore index for labels.

### API: `SignatureProgramDataset`
```python
class SignatureProgramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: list[SignatureProgramPair],
        tokenizer: Layer2Tokenizer,
        max_target_length: int = 16,
    ): ...
```

## Verification Plan

### Automated Tests
1. Create `tests/test_layer2_torch_dataset.py`.
2. Test `SignatureProgramDataset` construction and `__getitem__`.
3. Test that too-long programs are correctly filtered.
4. Test a `DataLoader` with the custom collation to verify:
   - `signature` shape: `(B, 12)`
   - `input_ids` shape: `(B, L)`
   - `labels` shape: `(B, L)`
   - `attention_mask` shape: `(B, L)`

### Manual Spot Check
Iterate through one batch and print shapes and sample values.

```python
from torch.utils.data import DataLoader
from eml_transformer.layer2.dataset import generate_pairs, GeneratorConfig
from eml_transformer.layer2.tokenizer import Layer2Tokenizer
from eml_transformer.layer2.torch_dataset import SignatureProgramDataset, collate_signature_fn
from eml_transformer.compiler.catalog import load_catalog

pairs = generate_pairs(GeneratorConfig(max_composition_depth=1))
tok = Layer2Tokenizer.from_catalog(load_catalog())
ds = SignatureProgramDataset(pairs, tok)
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_signature_fn(tok.pad_id))

batch = next(iter(dl))
for k, v in batch.items():
    print(f"{k:15s}: {v.shape}")
```
