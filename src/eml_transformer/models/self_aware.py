"""Main EML Transformer integrating the Effort Evaluator and Self-Aware layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.effort_head import EffortHead
from eml_transformer.training import load_checkpoint


class EMLTransformer(nn.Module):
    """Transformer that modulates its own layers based on predicted effort.

    This model encapsulates:
    1. A frozen 'Effort Evaluator' (TinyDecoder + EffortHead).
    2. A 'Main Decoder' with Self-Aware layers.

    The Effort Evaluator predicts per-token computational complexity, which
    is then used to scale the learnable 'delta' weights in the Main Decoder's
    FFN sublayers.
    """

    def __init__(
        self,
        evaluator_decoder: TinyDecoder,
        evaluator_head: EffortHead,
        main_decoder: TinyDecoder,
        task_head: nn.Module | None = None,
        freeze_evaluator: bool = True,
    ) -> None:
        super().__init__()
        self.evaluator_decoder = evaluator_decoder
        self.evaluator_head = evaluator_head
        self.main_decoder = main_decoder
        self.task_head = task_head

        if freeze_evaluator:
            for p in self.evaluator_decoder.parameters():
                p.requires_grad = False
            for p in self.evaluator_head.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) padding mask.

        Returns:
            (batch, seq_len, d_model) final hidden states from main decoder.
        """
        # 1. Predict effort (usually without gradients to save memory)
        # We use a context manager to ensure no grad if frozen.
        if not next(self.evaluator_decoder.parameters()).requires_grad:
            with torch.no_grad():
                eval_hidden = self.evaluator_decoder(
                    input_ids, attention_mask=attention_mask
                )
                effort = self.evaluator_head.effort_scalar(eval_hidden).unsqueeze(-1)
        else:
            eval_hidden = self.evaluator_decoder(
                input_ids, attention_mask=attention_mask
            )
            effort = self.evaluator_head.effort_scalar(eval_hidden).unsqueeze(-1)

        # 2. Modulate and compute main output
        hidden = self.main_decoder(
            input_ids, attention_mask=attention_mask, effort=effort
        )
        if self.task_head is not None:
            return self.task_head(hidden)
        return hidden

    @classmethod
    def from_checkpoints(
        cls,
        evaluator_path: str,
        main_config: ModelConfig,
        device: str = "cpu",
    ) -> EMLTransformer:
        """Factory method to build an EMLTransformer from a saved evaluator."""

        # 1. Load the evaluator config first.
        # We need a dummy model/head just to call load_checkpoint if we want
        # to use the existing utility, or just torch.load directly.
        # Since load_checkpoint returns the config, we can use it.

        # Temporary load to get config size
        temp_ckpt = torch.load(evaluator_path, map_location="cpu", weights_only=True)
        eval_config_data = temp_ckpt.get("config")
        if isinstance(eval_config_data, dict):
            eval_config = ModelConfig(**eval_config_data)
        elif eval_config_data is None:
            # Fallback to defaults if no config in checkpoint
            eval_config = ModelConfig()
        else:
            eval_config = eval_config_data
        eval_decoder = TinyDecoder(eval_config)
        eval_head = EffortHead(
            d_model=eval_config.d_model, num_bins=eval_config.num_bins
        )

        load_checkpoint(eval_decoder, eval_head, evaluator_path, device=device)

        main_decoder = TinyDecoder(main_config)
        return cls(eval_decoder, eval_head, main_decoder)
