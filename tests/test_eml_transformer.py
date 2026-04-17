import pytest
import torch

from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.effort_head import EffortHead
from eml_transformer.models.self_aware import EMLTransformer


def test_eml_transformer_forward():
    """Verify full EMLTransformer forward pass (evaluator -> effort -> main)."""
    # 1. Setup evaluator
    eval_config = ModelConfig(vocab_size=10, d_model=32, n_heads=2, n_layers=2)
    eval_decoder = TinyDecoder(eval_config)
    eval_head = EffortHead(d_model=32, num_bins=6)

    # 2. Setup main decoder (self-aware)
    main_config = ModelConfig(
        vocab_size=10, d_model=32, n_heads=2, n_layers=2, self_aware=True
    )
    main_decoder = TinyDecoder(main_config)

    # 3. Assemble
    model = EMLTransformer(eval_decoder, eval_head, main_decoder)

    # 4. Forward
    input_ids = torch.randint(0, 10, (2, 8))
    out = model(input_ids)

    assert out.shape == (2, 8, 32)


def test_eml_transformer_frozen_evaluator():
    """Verify that evaluator parameters don't require grad by default."""
    eval_config = ModelConfig(d_model=32)
    eval_decoder = TinyDecoder(eval_config)
    eval_head = EffortHead(d_model=32, num_bins=6)
    main_decoder = TinyDecoder(eval_config)

    model = EMLTransformer(eval_decoder, eval_head, main_decoder, freeze_evaluator=True)

    for p in model.evaluator_decoder.parameters():
        assert not p.requires_grad
    for p in model.evaluator_head.parameters():
        assert not p.requires_grad
    for p in model.main_decoder.parameters():
        assert p.requires_grad
