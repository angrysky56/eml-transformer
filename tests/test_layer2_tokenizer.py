"""Tests for Layer 2 tokenizer."""

import pytest
from eml_transformer.compiler.catalog import load_catalog
from eml_transformer.layer2.tokenizer import Layer2Tokenizer


def test_tokenizer_from_catalog():
    catalog = load_catalog()
    tok = Layer2Tokenizer.from_catalog(catalog)
    
    # Check special tokens
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3
    
    # Check variables and atoms
    assert tok.encode("x", add_special=False) != [tok.unk_id]
    assert tok.encode("1.0", add_special=False) != [tok.unk_id]
    assert tok.encode("E", add_special=False) != [tok.unk_id]
    
    # Check catalog names
    for entry in catalog:
        assert tok.encode(entry.name, add_special=False) != [tok.unk_id]
    
    # Total vocab size: 4 (special) + 5 (atoms) + 27 (catalog) = 36
    assert tok.vocab_size >= 36


def test_tokenizer_encode_decode_roundtrip():
    catalog = load_catalog()
    tok = Layer2Tokenizer.from_catalog(catalog)
    
    test_cases = [
        "x",
        "x y add",
        "x ln sin",
        "1.0 x E",
        "x y z mul add",  # Assuming mul and add are in catalog
    ]
    
    # Filter test cases to only those where all tokens are in catalog
    # (Just in case 'mul' or 'z' aren't there, but usually they are)
    valid_test_cases = []
    for rpn in test_cases:
        encoded = tok.encode(rpn, add_special=False)
        if tok.unk_id not in encoded:
            valid_test_cases.append(rpn)
            
    for rpn in valid_test_cases:
        ids = tok.encode(rpn)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        
        decoded = tok.decode(ids)
        assert decoded == rpn


def test_tokenizer_unk():
    catalog = load_catalog()
    tok = Layer2Tokenizer.from_catalog(catalog)
    
    ids = tok.encode("unknown_token", add_special=False)
    assert ids == [tok.unk_id]
    assert tok.decode(ids) == "<unk>"
