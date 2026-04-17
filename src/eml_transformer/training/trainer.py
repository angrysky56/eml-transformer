"""Training loop and configuration for the Effort Evaluator."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from eml_transformer.models.decoder import ModelConfig

torch.serialization.add_safe_globals([ModelConfig])

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
    """Main training loop for predicting EML tree depth."""
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
            hidden = model(input_ids)
            logits = head(hidden)

            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if config.verbose:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{config.epochs} | loss: {avg_loss:.4f} | time: {elapsed:.2f}s"
            )
            eval_metrics = evaluate(model, head, eval_loader, config.device)
            pretty_print_metrics(eval_metrics, prefix=f"Epoch {epoch} Eval")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> DepthMetrics:
    """Evaluate model on depth prediction."""
    model.eval()
    head.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["depth_labels"].to(device)
        hidden = model(input_ids)
        preds = head.predict_depth(hidden)
        all_preds.append(preds.view(-1).cpu())
        all_targets.append(targets.view(-1).cpu())
    if not all_preds:
        return compute_metrics(torch.tensor([]), torch.tensor([]))
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))


def train_lm(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: TrainConfig,
) -> None:
    """Train the model on the Next Token Prediction task."""
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=DEPTH_IGNORE_INDEX)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.device)
            targets = batch["lm_labels"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)

            # Flatten: (B*T, C) and (B*T,)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if config.verbose:
            elapsed = time.time() - start_time
            acc = evaluate_lm(model, eval_loader, config.device)
            print(
                f"Epoch {epoch}/{config.epochs} | loss: {avg_loss:.4f} | acc: {acc:.2%} | time: {elapsed:.2f}s"
            )


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Evaluate Next Token Prediction accuracy."""
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["lm_labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)

        mask = targets != DEPTH_IGNORE_INDEX
        correct += (preds[mask] == targets[mask]).sum().item()
        total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def save_checkpoint(
    model: nn.Module,
    head: nn.Module,
    config: Any,
    path: str,
) -> None:
    """Save model, head state dicts, and config to a single file."""
    # If config is a dataclass, convert to dict for safe serialization
    config_save = asdict(config) if is_dataclass(config) else config
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "config": config_save,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    head: nn.Module,
    path: str,
    device: str = "cpu",
) -> Any:
    """Load model and head state dicts and return the saved config."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    model.to(device)
    head.to(device)
    print(f"Checkpoint loaded from {path} (moved to {device})")
    return checkpoint.get("config")
