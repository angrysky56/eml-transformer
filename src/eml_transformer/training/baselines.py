"""Dumb baselines for the Effort Evaluator.

Any real model has to beat these to count as having learned something
beyond trivial statistics. If the transformer's accuracy isn't meaningfully
above the token-class baseline, the signal isn't learnable with local
context — which would itself be an important finding to stop and rethink.

Baselines computed here:
  1. GlobalMeanBaseline: predict the rounded mean depth everywhere.
  2. TokenClassBaseline: predict the mean depth conditional on token class
     (E / leaf / reserved). Exploits the structural fact that leaves are
     always depth 0; internal E nodes have varied depth.

Both baselines are fit from training data, evaluated on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eml_transformer.data.dataset import DEPTH_IGNORE_INDEX
from eml_transformer.data.tokenizer import EMLTokenizer


@dataclass
class GlobalMeanBaseline:
    """Predict a single constant depth for every valid position.

    Fit chooses the integer prediction minimizing mean absolute error on
    training data, which is simply the median — but for proof-of-life we
    use rounded mean for interpretability.
    """

    prediction: int = 0

    @classmethod
    def fit(cls, loader: DataLoader) -> GlobalMeanBaseline:
        total = 0.0
        count = 0
        for batch in loader:
            labels = batch["depth_labels"]
            valid = labels != DEPTH_IGNORE_INDEX
            total += labels[valid].float().sum().item()
            count += int(valid.sum().item())
        if count == 0:
            return cls(prediction=0)
        return cls(prediction=int(round(total / count)))

    def predict(self, input_ids: Tensor) -> Tensor:
        """Return a constant prediction tensor with the same shape as input_ids."""
        return torch.full_like(input_ids, fill_value=self.prediction)


@dataclass
class TokenClassBaseline:
    """Predict depth conditional on token class.

    Three classes: `E` (internal EML op), leaf (constant 1 or variable x/y/...),
    and `reserved` (bos/eos/pad/unk). Reserved positions are never supervised,
    so their prediction doesn't matter for metrics but we still emit zero
    there for tensor shape convenience.

    The per-class prediction is the rounded mean of true depths across all
    training positions of that class.
    """

    e_prediction: int = 0
    leaf_prediction: int = 0
    eml_id: int = -1
    const_id: int = -1
    variable_ids: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def fit(cls, loader: DataLoader, tokenizer: EMLTokenizer) -> TokenClassBaseline:
        from eml_transformer.data.tokenizer import (
            CONST_ONE_TOKEN,
            EML_OPCODE_TOKEN,
            RESERVED_TOKENS,
        )

        eml_id = tokenizer.token_to_id[EML_OPCODE_TOKEN]
        const_id = tokenizer.token_to_id[CONST_ONE_TOKEN]
        reserved = {tokenizer.token_to_id[t] for t in RESERVED_TOKENS}
        # Any non-reserved, non-E, non-const id is a variable.
        variable_ids = tuple(
            i
            for i in range(tokenizer.vocab_size)
            if i not in reserved and i != eml_id and i != const_id
        )

        e_total, e_count = 0.0, 0
        leaf_total, leaf_count = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"]
            labels = batch["depth_labels"]
            valid = labels != DEPTH_IGNORE_INDEX
            e_mask = valid & (input_ids == eml_id)
            leaf_mask = valid & (
                (input_ids == const_id)
                | torch.isin(
                    input_ids, torch.tensor(variable_ids, dtype=input_ids.dtype)
                )
            )
            e_total += labels[e_mask].float().sum().item()
            e_count += int(e_mask.sum().item())
            leaf_total += labels[leaf_mask].float().sum().item()
            leaf_count += int(leaf_mask.sum().item())

        e_pred = int(round(e_total / e_count)) if e_count else 0
        leaf_pred = int(round(leaf_total / leaf_count)) if leaf_count else 0
        return cls(
            e_prediction=e_pred,
            leaf_prediction=leaf_pred,
            eml_id=eml_id,
            const_id=const_id,
            variable_ids=variable_ids,
        )

    def predict(self, input_ids: Tensor) -> Tensor:
        """Per-token predictions based on token class.

        Shape matches `input_ids`. Reserved / unknown positions default to 0,
        but the evaluator should mask those out anyway via DEPTH_IGNORE_INDEX.
        """
        preds = torch.zeros_like(input_ids)
        preds[input_ids == self.eml_id] = self.e_prediction
        # Leaves: const_one or any variable.
        leaf_mask = input_ids == self.const_id
        if self.variable_ids:
            var_tensor = torch.tensor(
                self.variable_ids, device=input_ids.device, dtype=input_ids.dtype
            )
            leaf_mask = leaf_mask | torch.isin(input_ids, var_tensor)
        preds[leaf_mask] = self.leaf_prediction
        return preds
