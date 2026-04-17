"""Tests for the EffortDataset and collate function."""

from __future__ import annotations

import torch

from eml_transformer.data.dataset import (
    DEPTH_IGNORE_INDEX,
    EffortDataset,
    collate_effort_batch,
)
from eml_transformer.data.tokenizer import EMLTokenizer


def _build_dataset(num_samples: int = 8, max_depth: int = 4, seed: int = 0) -> EffortDataset:
    tokenizer = EMLTokenizer.from_variables(["x", "y"])
    return EffortDataset(
        tokenizer=tokenizer,
        variables=["x", "y"],
        num_samples=num_samples,
        max_depth=max_depth,
        seed=seed,
    )


def test_dataset_len_matches_num_samples() -> None:
    dataset = _build_dataset(num_samples=12)
    assert len(dataset) == 12


def test_sample_has_parallel_ids_and_labels() -> None:
    dataset = _build_dataset()
    sample = dataset[0]
    assert len(sample.input_ids) == len(sample.depth_labels)


def test_bos_and_eos_label_is_ignore_index() -> None:
    """Structural tokens don't get depth labels — the loss must skip them."""
    dataset = _build_dataset()
    sample = dataset[0]
    assert sample.depth_labels[0] == DEPTH_IGNORE_INDEX
    assert sample.depth_labels[-1] == DEPTH_IGNORE_INDEX


def test_sample_determinism_across_reads() -> None:
    """Same (seed, index) must yield identical samples."""
    dataset_a = _build_dataset(seed=42)
    dataset_b = _build_dataset(seed=42)
    for i in range(len(dataset_a)):
        a = dataset_a[i]
        b = dataset_b[i]
        assert a.input_ids == b.input_ids
        assert a.depth_labels == b.depth_labels
        assert a.tokens == b.tokens


def test_collate_pads_to_longest_sequence() -> None:
    dataset = _build_dataset(num_samples=4)
    samples = [dataset[i] for i in range(4)]
    batch = collate_effort_batch(samples, pad_id=dataset.tokenizer.pad_id)
    max_len = max(len(s.input_ids) for s in samples)
    assert batch["input_ids"].shape == (4, max_len)
    assert batch["depth_labels"].shape == (4, max_len)
    assert batch["attention_mask"].shape == (4, max_len)


def test_collate_tensor_dtypes() -> None:
    dataset = _build_dataset(num_samples=4)
    samples = [dataset[i] for i in range(4)]
    batch = collate_effort_batch(samples, pad_id=dataset.tokenizer.pad_id)
    assert batch["input_ids"].dtype == torch.long
    assert batch["depth_labels"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool


def test_collate_padding_positions_use_ignore_label() -> None:
    """Padded positions in the label tensor must equal DEPTH_IGNORE_INDEX so
    masked losses (e.g., CrossEntropyLoss(ignore_index=-100)) skip them."""
    dataset = _build_dataset(num_samples=6)
    samples = [dataset[i] for i in range(6)]
    batch = collate_effort_batch(samples, pad_id=dataset.tokenizer.pad_id)
    # Any position where attention_mask is False must have the ignore label.
    ignore_positions = ~batch["attention_mask"]
    if ignore_positions.any():
        assert (batch["depth_labels"][ignore_positions] == DEPTH_IGNORE_INDEX).all()
