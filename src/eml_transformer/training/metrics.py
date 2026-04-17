"""Evaluation metrics for the Effort Evaluator.

All metrics respect the `DEPTH_IGNORE_INDEX` convention: positions where
the label equals the ignore index are excluded from counts and sums.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from eml_transformer.data.dataset import DEPTH_IGNORE_INDEX


@dataclass
class DepthMetrics:
    """Summary of effort-prediction quality on a dataset.

    `accuracy` is exact-match rate; `mae` is mean absolute error. Both are
    computed over valid (non-ignored) positions only. `per_depth_accuracy`
    is a dict mapping true-depth value to (accuracy, count) for fine-grained
    inspection of where the model fails.
    """

    accuracy: float
    mae: float
    total_positions: int
    per_depth_accuracy: dict[int, tuple[float, int]]


def compute_metrics(preds: Tensor, targets: Tensor) -> DepthMetrics:
    """Calculate accuracy and MAE between predictions and targets.

    Targets equal to DEPTH_IGNORE_INDEX are excluded.
    `preds` can be (B, T) discrete predictions or raw values; they will be
    compared directly to targets.
    """
    valid_mask = targets != DEPTH_IGNORE_INDEX
    valid_preds = preds[valid_mask].float()
    valid_targets = targets[valid_mask].float()

    if valid_targets.numel() == 0:
        return DepthMetrics(0.0, 0.0, 0, {})

    accuracy = (valid_preds.round() == valid_targets).float().mean().item()
    mae = torch.abs(valid_preds - valid_targets).mean().item()
    total_positions = int(valid_targets.numel())

    per_depth_accuracy = {}
    unique_depths = torch.unique(valid_targets)
    for d in unique_depths:
        d_int = int(d.item())
        d_mask = valid_targets == d
        d_preds = valid_preds[d_mask]
        d_acc = (d_preds.round() == d).float().mean().item()
        d_count = int(d_mask.sum().item())
        per_depth_accuracy[d_int] = (d_acc, d_count)

    return DepthMetrics(
        accuracy=accuracy,
        mae=mae,
        total_positions=total_positions,
        per_depth_accuracy=per_depth_accuracy,
    )


def pretty_print_metrics(metrics: DepthMetrics, prefix: str = "") -> None:
    """Format metrics for console output."""
    if prefix:
        print(f"\n--- {prefix} Metrics ---")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"MAE:      {metrics.mae:.4f}")
    print(f"Samples:  {metrics.total_positions}")
    print("Per-depth accuracy:")
    for d in sorted(metrics.per_depth_accuracy.keys()):
        acc, count = metrics.per_depth_accuracy[d]
        print(f"  depth {d}: {acc:.2%} (n={count})")
