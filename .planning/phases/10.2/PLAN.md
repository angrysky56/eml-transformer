# Phase 10.2: Layer 2 Tokenizer

This phase implements a dedicated tokenizer for Layer 2. Unlike the Layer 1 tokenizer which works at the character/token level for EML primitives, this tokenizer treats catalog entries as atomic tokens.

## Objectives
- Implement `src/eml_transformer/layer2/tokenizer.py`.
- Define a fixed-vocabulary tokenizer where catalog names are atoms.
- Include special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`), variables (`x`, `y`, `z`), and specific literals (`1.0`, `E`).
- Support round-trip encoding and decoding.

## Requirements

### Vocabulary Specification
- `<pad>` = 0
- `<bos>` = 1
- `<eos>` = 2
- `<unk>` = 3
- Variables: `x`, `y`, `z`
- Literal: `1.0`
- Operator: `E`
- Catalog Names: All entries from `load_catalog()` as single tokens.

### Class: `Layer2Tokenizer`
- `from_catalog(catalog: list[CatalogEntry]) -> Layer2Tokenizer`: Factory method to build vocab from catalog.
- `encode(rpn: str, add_special: bool = True) -> list[int]`: Convert RPN string to token IDs.
- `decode(ids: list[int], strip_special: bool = True) -> str`: Convert token IDs back to RPN string.
- Properties for special token IDs.

## Verification Plan

### Automated Tests
1. Create `tests/test_layer2_tokenizer.py`.
2. Test `from_catalog` correctly includes all catalog names.
3. Test `encode`/`decode` round-trip for:
   - Simple variable: `"x"`
   - Catalog call: `"x sin"`
   - Binary op: `"x y add"`
   - Literal: `"1.0 x E"`
4. Verify special tokens are handled correctly (BOS/EOS).

### Manual Spot Check
Run a script to print the vocabulary size and a few round-trip examples.

```python
from eml_transformer.compiler.catalog import load_catalog
from eml_transformer.layer2.tokenizer import Layer2Tokenizer

tok = Layer2Tokenizer.from_catalog(load_catalog())
print(f"vocab_size = {tok.vocab_size}")
for rpn in ["x", "x y add", "x ln sin", "1.0 x E"]:
    encoded = tok.encode(rpn)
    decoded = tok.decode(encoded)
    print(f"{rpn:15s} -> {encoded} -> {decoded}")
    assert decoded == rpn
```
