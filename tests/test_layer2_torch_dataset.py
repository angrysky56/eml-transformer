"""Tests for Layer 2 PyTorch dataset and collation."""

import torch
import pytest
from torch.utils.data import DataLoader
from eml_transformer.compiler.catalog import load_catalog
from eml_transformer.layer2.dataset import SignatureProgramPair, GeneratorConfig, generate_pairs
from eml_transformer.layer2.tokenizer import Layer2Tokenizer
from eml_transformer.layer2.torch_dataset import SignatureProgramDataset, collate_signature_fn


def test_signature_program_dataset_basic():
    # Mock some data
    pairs = [
        SignatureProgramPair(signature=(1j, 2j, 3j, 4j, 5j, 6j), rpn="x"),
        SignatureProgramPair(signature=(0, 0, 0, 0, 0, 0), rpn="x sin"),
    ]
    tok = Layer2Tokenizer.from_catalog(load_catalog())
    ds = SignatureProgramDataset(pairs, tok, max_target_length=10)
    
    assert len(ds) == 2
    item = ds[0]
    assert "signature" in item
    assert "ids" in item
    assert item["signature"].shape == (12,)
    # x -> [BOS, x, EOS] -> length 3
    assert item["ids"].shape == (3,)
    
    # Check signature values: [r0, i0, r1, i1, ...]
    # 1j -> [0, 1]
    assert item["signature"][0] == 0
    assert item["signature"][1] == 1


def test_dataset_filtering():
    # Test that too-long programs are dropped
    pairs = [
        SignatureProgramPair(signature=(0,)*6, rpn="x"), # length 3
        SignatureProgramPair(signature=(0,)*6, rpn="x " * 10 + "sin"), # length 13
    ]
    tok = Layer2Tokenizer.from_catalog(load_catalog())
    ds = SignatureProgramDataset(pairs, tok, max_target_length=10)
    
    assert len(ds) == 1
    assert ds[0]["ids"].shape[0] <= 10


def test_collate_signature():
    tok = Layer2Tokenizer.from_catalog(load_catalog())
    pairs = [
        SignatureProgramPair(signature=(1j,)*6, rpn="x"),       # ids length 3
        SignatureProgramPair(signature=(2j,)*6, rpn="x sin"),   # ids length 4
    ]
    ds = SignatureProgramDataset(pairs, tok)
    collate_fn = collate_signature_fn(tok.pad_id)
    
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))
    
    # batch_max_len = 4
    # sequence lengths: 3, 4
    # full_padded (L=4):
    # [PAD, BOS, x,   EOS]
    # [BOS, x,   sin, EOS]
    
    # input_ids (L-1 = 3):
    # [PAD, BOS, x]
    # [BOS, x,   sin]
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][0, 0] == tok.pad_id
    assert batch["input_ids"][0, 1] == tok.bos_id
    assert batch["input_ids"][1, 0] == tok.bos_id
    
    # labels (L-1 = 3):
    # [BOS, x,   EOS] -> then mask PAD -> [-100, x, EOS]
    # [x,   sin, EOS]
    assert batch["labels"].shape == (2, 3)
    assert batch["labels"][0, 0] == -100
    assert batch["labels"][0, 1] == tok.encode("x", add_special=False)[0]
    assert batch["labels"][0, 2] == tok.eos_id
    
    # attention_mask
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["attention_mask"][0, 0] == False
    assert batch["attention_mask"][0, 1] == True
    
    # signature
    assert batch["signature"].shape == (2, 12)
    assert batch["signature"][0, 1] == 1.0
    assert batch["signature"][1, 1] == 2.0


def test_dataloader_integration_smoke():
    # Real data smoke test
    cfg = GeneratorConfig(max_composition_depth=1)
    pairs = generate_pairs(cfg)
    tok = Layer2Tokenizer.from_catalog(load_catalog())
    ds = SignatureProgramDataset(pairs, tok)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_signature_fn(tok.pad_id))
    
    batch = next(iter(dl))
    assert "signature" in batch
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    
    B = batch["input_ids"].shape[0]
    L = batch["input_ids"].shape[1]
    assert batch["labels"].shape == (B, L)
    assert batch["attention_mask"].shape == (B, L)
    assert batch["signature"].shape == (B, 12)
