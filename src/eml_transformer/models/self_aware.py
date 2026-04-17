"""Composed EML Transformer: frozen evaluator feeding a main decoder.

The ``EMLTransformer`` wires together two ``TinyDecoder`` instances:

* An **evaluator** (decoder + ``EffortHead``) that produces a per-token
  effort scalar in ``[0, 1]``. Typically loaded from a checkpoint and frozen.
* A **main decoder** whose FFNs consume the effort scalar via FiLM or delta
  modulation. This is the model being trained at Phase 2.

Two notable design choices:

1. The evaluator is called under ``torch.no_grad()`` when frozen — no
   gradients flow back through it even if autograd is enabled, which is
   both faster and safer.
2. ``from_checkpoints`` reads the checkpoint file exactly once (the
   earlier implementation read it twice through a utility indirection).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.effort_head import EffortHead


def _coerce_config(raw: Any) -> ModelConfig:
    """Turn a possibly-dict (or legacy) config payload into a ``ModelConfig``.

    Supports three shapes: a ``ModelConfig`` instance (pass through), a dict
    of fields (expanded via ``ModelConfig(**dict)`` through the legacy-aware
    ``make_config``), or None (fall back to defaults). Legacy ``self_aware``
    keys are translated to ``ffn_mode`` inside ``make_config``.
    """
    from eml_transformer.models.decoder import make_config

    if raw is None:
        return ModelConfig()
    if isinstance(raw, ModelConfig):
        return raw
    if isinstance(raw, dict):
        return make_config(**raw)
    raise TypeError(f"unsupported config type: {type(raw).__name__}")


class EMLTransformer(nn.Module):
    """Frozen evaluator + modulated main decoder + optional task head.

    The task head is pluggable (e.g. ``LMHead`` for next-token prediction).
    When ``task_head`` is ``None``, ``forward`` returns raw hidden states.
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
        self.evaluator_frozen = freeze_evaluator

        if freeze_evaluator:
            for p in self.evaluator_decoder.parameters():
                p.requires_grad = False
            for p in self.evaluator_head.parameters():
                p.requires_grad = False

    def _compute_effort(
        self, input_ids: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        """Run the evaluator and return (B, T, 1) effort scalars.

        Always returns a tensor shaped for broadcast into modulated FFNs.
        If the evaluator is frozen, runs under ``no_grad`` to skip autograd
        bookkeeping entirely.
        """
        if self.evaluator_frozen:
            with torch.no_grad():
                h = self.evaluator_decoder(input_ids, attention_mask=attention_mask)
                effort = self.evaluator_head.effort_scalar(h)
        else:
            h = self.evaluator_decoder(input_ids, attention_mask=attention_mask)
            effort = self.evaluator_head.effort_scalar(h)
        return effort.unsqueeze(-1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: evaluator -> main decoder -> optional task head."""
        # For vanilla main decoders, skip the evaluator entirely — it would be
        # wasted compute since the effort signal is ignored downstream.
        if self.main_decoder.config.ffn_mode == "vanilla":
            hidden = self.main_decoder(input_ids, attention_mask=attention_mask)
        else:
            effort = self._compute_effort(input_ids, attention_mask)
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
        freeze_evaluator: bool = True,
    ) -> "EMLTransformer":
        """Build an ``EMLTransformer`` by loading an evaluator from disk.

        Single ``torch.load`` for both config *and* weights. Uses
        ``weights_only=False`` because the checkpoint may contain a
        ``ModelConfig`` dataclass — we trust our own checkpoints.
        """
        ckpt = torch.load(evaluator_path, map_location=device, weights_only=False)
        eval_config = _coerce_config(ckpt.get("config"))

        eval_decoder = TinyDecoder(eval_config)
        eval_head = EffortHead(
            d_model=eval_config.d_model, num_bins=eval_config.num_bins
        )
        eval_decoder.load_state_dict(ckpt["model_state_dict"])
        eval_head.load_state_dict(ckpt["head_state_dict"])

        main_decoder = TinyDecoder(main_config)
        return cls(
            evaluator_decoder=eval_decoder,
            evaluator_head=eval_head,
            main_decoder=main_decoder,
            freeze_evaluator=freeze_evaluator,
        )

    def num_trainable_parameters(self) -> int:
        """Parameters that actually receive gradients — excludes frozen evaluator.

        This is the number to cite when comparing Phase-2 models. It is *not*
        the same as ``main_decoder.num_parameters()`` when the task head exists.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
