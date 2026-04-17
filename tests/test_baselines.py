"""Verification tests for baseline models."""

import torch
from torch.utils.data import DataLoader
from eml_transformer.training.baselines import GlobalMeanBaseline, TokenClassBaseline
from eml_transformer.data.tokenizer import EMLTokenizer
from eml_transformer.data.dataset import EffortDataset, collate_effort_batch


def test_global_mean_baseline_fit_predict():
    """GlobalMeanBaseline must predict the same constant everywhere."""
    tokenizer = EMLTokenizer.from_variables(["x"])
    ds = EffortDataset(tokenizer, ["x"], num_samples=10, seed=42)
    loader = DataLoader(
        ds, batch_size=2, collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id)
    )

    baseline = GlobalMeanBaseline.fit(loader)

    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 5))
    preds = baseline.predict(input_ids)

    assert preds.shape == (2, 5)
    assert preds.dtype == torch.long
    assert (preds == baseline.prediction).all()


def test_token_class_baseline_fit_predict():
    """TokenClassBaseline must differentiate between internal and leaf nodes."""
    tokenizer = EMLTokenizer.from_variables(["x", "y"])
    ds = EffortDataset(tokenizer, ["x", "y"], num_samples=10, seed=42)
    loader = DataLoader(
        ds, batch_size=2, collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id)
    )

    baseline = TokenClassBaseline.fit(loader, tokenizer)

    # Manual input with one E token and one variable token.
    input_ids = torch.tensor([[tokenizer.token_to_id["E"], tokenizer.token_to_id["x"]]])
    preds = baseline.predict(input_ids)

    assert preds.shape == (1, 2)
    assert preds[0, 0] == baseline.e_prediction
    assert preds[0, 1] == baseline.leaf_prediction
