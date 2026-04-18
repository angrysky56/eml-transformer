"""Tests for Layer 2 training loop and metrics."""

import torch
from eml_transformer.layer2.train import get_call_count, evaluate
from eml_transformer.layer2.model import SignatureProgramModel, Layer2Config
from eml_transformer.layer2.tokenizer import Layer2Tokenizer
from eml_transformer.compiler import load_catalog
from torch.utils.data import DataLoader
from eml_transformer.layer2.dataset import SignatureProgramPair
from eml_transformer.layer2.torch_dataset import SignatureProgramDataset, collate_signature_fn


def test_get_call_count():
    assert get_call_count("x") == 0
    assert get_call_count("x y add") == 1
    assert get_call_count("x sin cos") == 2
    assert get_call_count("x y add z mul") == 2


def test_evaluate_smoke():
    # Setup tiny model and data
    catalog = load_catalog()
    tok = Layer2Tokenizer.from_catalog(catalog)
    cfg = Layer2Config(vocab_size=tok.vocab_size, d_model=32, n_decoder_layers=1)
    model = SignatureProgramModel(cfg)
    
    # Mock data: 2 samples
    pairs = [
        SignatureProgramPair(tuple([0j]*6), "x"),
        SignatureProgramPair(tuple([1j]*6), "x sin")
    ]
    ds = SignatureProgramDataset(pairs, tok)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_signature_fn(tok.pad_id))
    
    device = "cpu"
    loss, em, sm = evaluate(model, loader, tok, device)
    
    assert isinstance(loss, float)
    assert 0 <= em <= 1.0
    assert 0 <= sm <= 1.0
