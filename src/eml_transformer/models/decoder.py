"""Small decoder-only transformer body.

The decoder body is shared between two roles:

* **Effort Evaluator** — a decoder + ``EffortHead`` trained to predict
  per-token subtree depth on EML RPN sequences.
* **Main decoder** — the downstream model whose FFNs can be modulated by
  the evaluator's effort signal (via ``ffn_mode="film"`` or ``"delta"``).

``ModelConfig`` carries all the knobs. The evaluator only needs
``num_bins`` to size its classification head, and the main decoder only
needs ``ffn_mode`` to pick its FFN variant — the *other* field is inert
for the *other* role, which is intentional. Splitting them into two
dataclasses would require extra boilerplate without reducing confusion
once a reader knows which role they're configuring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from eml_transformer.models.layers import DecoderLayer

VALID_FFN_MODES: tuple[str, ...] = ("vanilla", "delta", "film")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the tiny decoder.

    Defaults target a ~800K-parameter evaluator suitable for the proof-of-life
    experiment. For main decoders that consume effort, set ``ffn_mode``.

    Args:
        vocab_size: Size of the tokenizer's output alphabet.
        d_model: Hidden dimension.
        n_heads: Number of attention heads; must divide ``d_model``.
        n_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length for rotary embeddings.
        rms_eps: RMSNorm epsilon.
        ffn_mode: FFN variant. ``vanilla`` ignores effort; ``delta`` duplicates
            the FFN; ``film`` applies FiLM modulation.
        ffn_expansion: Expansion ratio for FFN hidden dim. Use this to
            parameter-match ``vanilla`` against ``delta`` or ``film``
            (e.g. expansion=6 brings vanilla closer to a delta model at 4).
        num_bins: Only used by the evaluator role (for EffortHead). Irrelevant
            to main-decoder roles. Default 6 covers EML depths 0..5.

    Backward compatibility: the old ``self_aware: bool`` flag is accepted via
    ``make_config(self_aware=True)`` and translated to ``ffn_mode="delta"``.
    """

    vocab_size: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128
    rms_eps: float = 1e-5
    ffn_mode: str = "vanilla"
    ffn_expansion: int = 4
    num_bins: int = 6

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
            )
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if self.ffn_mode not in VALID_FFN_MODES:
            raise ValueError(f"ffn_mode {self.ffn_mode!r} not in {VALID_FFN_MODES}")
        if self.ffn_expansion < 1:
            raise ValueError("ffn_expansion must be >= 1")
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")


torch.serialization.add_safe_globals([ModelConfig])


def _legacy_self_aware_config(**kwargs) -> dict:
    """Translate a legacy ``self_aware=True/False`` kwarg into ``ffn_mode``.

    Preserves the pre-refactor CLI / checkpoint surface. Quietly silences any
    mention of ``self_aware`` after translation.
    """
    if "self_aware" in kwargs:
        legacy = kwargs.pop("self_aware")
        if legacy and kwargs.get("ffn_mode", "vanilla") == "vanilla":
            kwargs["ffn_mode"] = "delta"
    return kwargs


def make_config(**kwargs) -> ModelConfig:
    """Build a ``ModelConfig`` accepting both current and legacy kwargs."""
    return ModelConfig(**_legacy_self_aware_config(**kwargs))


class TinyDecoder(nn.Module):
    """Decoder-only transformer producing per-position hidden states.

    No language-modeling head is attached here — a pluggable ``task_head``
    lives on the composed ``EMLTransformer`` instead, so this class stays
    usable as either evaluator body or main-decoder body.
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
                    ffn_mode=config.ffn_mode,
                    ffn_expansion=config.ffn_expansion,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        """Small-model init: normal(std=0.02) for linears and embeddings.

        FiLM generators are initialized separately in ``FiLMFFN.__init__`` and
        deliberately not overwritten here — FiLM layers need their γ/β
        predictor to start at zero so the model degrades to vanilla at init.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                # Skip FiLM generators so their zero init survives.
                if getattr(module, "_is_film_gen", False):
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        effort: Tensor | None = None,
    ) -> Tensor:
        """Forward pass producing hidden states.

        Args:
            input_ids: (batch, seq_len) long tensor of token IDs.
            attention_mask: (batch, seq_len) bool, True for real tokens.
            effort: optional (batch, seq_len, 1) float tensor consumed by
                modulated FFN variants. Ignored by vanilla FFNs.
        """
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"input length {input_ids.size(1)} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=attention_mask, effort=effort)
        return self.final_norm(x)

    def num_parameters(self) -> int:
        """Count of trainable parameters — useful for parameter-matched ablations."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
