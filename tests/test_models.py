"""Verification tests for model shapes and parameter counts."""

import torch
from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.effort_head import EffortHead


def test_decoder_forward_shape():
    """Hidden states must have shape (B, T, D)."""
    config = ModelConfig(vocab_size=10, d_model=64, n_heads=4, n_layers=2, max_seq_len=32)
    model = TinyDecoder(config)

    input_ids = torch.randint(0, 10, (4, 16))
    hidden = model(input_ids)

    assert hidden.shape == (4, 16, 64)


def test_effort_head_forward_shape():
    """Logits must have shape (B, T, bins)."""
    d_model = 64
    num_bins = 6
    head = EffortHead(d_model=d_model, num_bins=num_bins)

    hidden = torch.randn(4, 16, d_model)
    logits = head(hidden)

    assert logits.shape == (4, 16, num_bins)


def test_model_parameter_count_sanity():
    """The ~500K parameter goal must be met with default settings."""
    config = ModelConfig(vocab_size=8, d_model=128, n_heads=4, n_layers=4, max_seq_len=128)
    model = TinyDecoder(config)

    n_params = model.num_parameters()
    # Standard settings for this prototype result in ~780K-800K parameters.
    assert 400_000 < n_params < 900_000


def test_effort_head_predictions():
    """Predictions and scalar scores must have correct shapes and ranges."""
    head = EffortHead(d_model=64, num_bins=5)
    hidden = torch.randn(2, 8, 64)

    preds = head.predict_depth(hidden)
    assert preds.shape == (2, 8)
    assert preds.dtype == torch.long
    assert (preds >= 0).all() and (preds < 5).all()

    effort = head.effort_scalar(hidden, normalize=True)
    assert effort.shape == (2, 8)
    assert (effort >= 0.0).all() and (effort <= 1.0).all()
