"""Training loop and configuration for the Effort Evaluator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from eml_transformer.data.dataset import DEPTH_IGNORE_INDEX
from eml_transformer.training.metrics import compute_metrics, pretty_print_metrics

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from eml_transformer.training.metrics import DepthMetrics


@dataclass
class TrainConfig:
    """Hyperparameters for training."""

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    device: str = "cpu"
    verbose: bool = True


def train(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: TrainConfig,
) -> None:
    """Main training loop for predicting EML tree depth.

    Uses AdamW optimizer and CosineAnnealingLR scheduler.
    Loss is CrossEntropyLoss over discrete depth bins.
    """
    model.to(config.device)
    head.to(config.device)

    params = list(model.parameters()) + list(head.parameters())
    optimizer = AdamW(
        params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=DEPTH_IGNORE_INDEX)

    for epoch in range(1, config.epochs + 1):
        model.train()
        head.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.device)
            targets = batch["depth_labels"].to(config.device)

            optimizer.zero_grad()

            # Forward pass
            hidden = model(input_ids)  # (B, T, d_model)
            logits = head(hidden)      # (B, T, num_bins)

            # Flatten for CrossEntropyLoss: (B*T, C) and (B*T,)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if config.verbose:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{config.epochs} | loss: {avg_loss:.4f} | time: {elapsed:.2f}s")

            # Periodic evaluation
            eval_metrics = evaluate(model, head, eval_loader, config.device)
            pretty_print_metrics(eval_metrics, prefix=f"Epoch {epoch} Eval")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> DepthMetrics:
    """Evaluate model on a dataloader."""
    model.eval()
    head.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["depth_labels"].to(device)

        hidden = model(input_ids)
        # Assuming EffortHead has predict_depth method as seen in effort_head.py
        preds = head.predict_depth(hidden)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    return compute_metrics(preds_tensor, targets_tensor)
