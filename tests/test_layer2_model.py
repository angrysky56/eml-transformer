"""Tests for Layer 2 model architecture."""

import torch
import pytest
from eml_transformer.layer2.model import SignatureProgramModel, Layer2Config


def test_model_param_count():
    # Target: 0.5M - 1.0M for d_model=128, 4 layers
    # Our vocab size is ~36
    cfg = Layer2Config(vocab_size=36, d_model=128, n_decoder_layers=4)
    model = SignatureProgramModel(cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    # Estimated: ~1.07M. Let's see actual.
    assert 500_000 <= total_params <= 1_200_000


def test_model_forward_shape():
    cfg = Layer2Config(vocab_size=36, d_model=64, n_decoder_layers=2) # smaller for speed
    model = SignatureProgramModel(cfg)
    
    B, T = 2, 8
    sig = torch.randn(B, 12)
    ids = torch.randint(0, 36, (B, T))
    mask = torch.ones((B, T), dtype=torch.bool)
    
    logits = model(sig, ids, attention_mask=mask)
    assert logits.shape == (B, T, 36)


def test_model_generation():
    cfg = Layer2Config(vocab_size=36, d_model=64, n_decoder_layers=2)
    model = SignatureProgramModel(cfg)
    model.eval()
    
    B = 2
    sig = torch.randn(B, 12)
    
    # Greedy generation
    results = model.generate(sig, max_length=10, bos_id=1, eos_id=2, temperature=0.0)
    
    assert len(results) == B
    for res in results:
        assert isinstance(res, list)
        assert len(res) <= 10
        # Check that IDs are within vocab
        for token_id in res:
            assert 0 <= token_id < 36


def test_gradient_flow():
    cfg = Layer2Config(vocab_size=36, d_model=64, n_decoder_layers=2)
    model = SignatureProgramModel(cfg)
    
    sig = torch.randn(2, 12, requires_grad=True)
    ids = torch.randint(4, 36, (2, 8))
    
    logits = model(sig, ids)
    loss = logits.sum()
    loss.backward()
    
    # Check if signature gets gradients (encoder is working)
    assert sig.grad is not None
    assert sig.grad.abs().sum() > 0
    
    # Check if model weights have gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert p.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"
