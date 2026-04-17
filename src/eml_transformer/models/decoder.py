"""Small decoder-only transformer body for the Effort Evaluator.

This module defines `TinyDecoder` — a stack of RMSNorm + causal self-attention
+ FFN layers that maps token IDs to per-position hidden states. The effort
regression head in `effort_head.py` consumes those hidden states.

Configuration is a plain dataclass with sensible defaults for a ~500K
parameter model that trains in under a minute on a single RTX 3060.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from eml_transformer.models.layers import DecoderLayer


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the tiny decoder.

    Defaults target the proof-of-life setting: ~500K parameters trainable
    on a 12 GB GPU with batch size 64 in well under a minute per 4k steps.
    """

    vocab_size: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128
    rms_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
            )
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")


class TinyDecoder(nn.Module):
    """Decoder-only transformer producing per-position hidden states.

    No language-modeling head is attached here — the effort head is the
    sole output of this phase, and we want the decoder body to be reusable
    if later phases add auxiliary objectives (e.g., an LM head for joint
    pretraining on mixed language + EML corpora).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                    eps=config.rms_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        """Small-model init: normal(std=0.02) for linears and embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass producing hidden states.

        Args:
            input_ids: (batch, seq_len) long tensor of token IDs.
            attention_mask: (batch, seq_len) bool tensor with True for real
                tokens and False for padding. Passed to each decoder layer.

        Returns:
            (batch, seq_len, d_model) final hidden states.
        """
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"input length {input_ids.size(1)} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=attention_mask)
        return self.final_norm(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
